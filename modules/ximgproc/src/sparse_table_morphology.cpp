// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include <math.h>
#include <vector>
#include <iostream>
#include <stack>
#include <algorithm>
#include <queue>

namespace cv {
namespace ximgproc {
namespace stMorph {

// normalizeAnchor; Copied from filterengine.hpp.
static inline Point normalizeAnchor(Point anchor, Size ksize)
{
    if (anchor.x == -1)
        anchor.x = ksize.width / 2;
    if (anchor.y == -1)
        anchor.y = ksize.height / 2;
    CV_Assert(anchor.inside(Rect(0, 0, ksize.width, ksize.height)));
    return anchor;
}

enum Dim
{
    Col, Row
};

struct StStep
{
    StStep(int dimR, int dimC, Dim _ax)
    {
        dimRow = dimR;
        dimCol = dimC;
        ax = _ax;
    }
    int dimRow;
    int dimCol;
    Dim ax;
};

/*
* Find a smaller set of power-of-2 rectangles to cover the kernel.
* - The width and the height of each rectangles are power of 2.
* - Overlappings of rectangles are allowed.
*/
std::vector<Rect> genPow2RectsToCoverKernel(InputArray _kernel)
{
    CV_Assert(_kernel.type() == CV_8UC1);

    Mat kernel = _kernel.getMat();

    // generate log2 table
    int len = max(kernel.rows, kernel.cols) + 1;
    std::vector<int> log2(len);
    for (int i = 2; i < len; i++) log2[i] = log2[i >> 1] + 1;

    // generate sparse table for the kernel
    std::vector<std::vector<Mat>> st(log2[kernel.rows] + 1, std::vector<Mat>(log2[kernel.cols] + 1));
    st[0][0] = kernel;
    for (int colDepth = 1; colDepth <= log2[kernel.cols]; colDepth++)
    {
        int rowStep = 0;
        int rowSkip = 0;
        int rowLim = kernel.rows - rowSkip;

        int colStep = 1 << (colDepth - 1);
        int colSkip = (1 << colDepth) - 1;
        int colLim = kernel.cols - colSkip;

        st[0][colDepth] = Mat::zeros(kernel.rows, kernel.cols, kernel.type());
        uchar* ptr1 = st[0][colDepth - 1].ptr();
        uchar* ptr2 = st[0][colDepth - 1].ptr(rowStep, colStep);
        uchar* dst = st[0][colDepth].ptr();
        for (int row = 0; row < rowLim; row++)
        {
            for (int col = 0; col < colLim; col++)
            {
                *dst++ = *ptr1++ & *ptr2++;
            }
            ptr1 += colSkip;
            ptr2 += colSkip;
            dst += colSkip;
        }
    }
    for (int rowDepth = 1; rowDepth <= log2[kernel.rows]; rowDepth++)
    {
        int rowStep = 1 << (rowDepth - 1);
        int rowSkip = (1 << rowDepth) - 1;
        int rowLim = kernel.rows - rowSkip;
        for (int colDepth = 0; colDepth <= log2[kernel.cols]; colDepth++)
        {
            int colStep = 0;
            int colSkip = (1 << colDepth) - 1;
            int colLim = kernel.cols - colSkip;

            st[rowDepth][colDepth] = Mat::zeros(kernel.rows, kernel.cols, kernel.type());
            uchar* ptr1 = st[rowDepth - 1][colDepth].ptr();
            uchar* ptr2 = st[rowDepth - 1][colDepth].ptr(rowStep, colStep);
            uchar* dst = st[rowDepth][colDepth].ptr();
            for (int row = 0; row < rowLim; row++)
            {
                for (int col = 0; col < colLim; col++)
                {
                    *dst++ = *ptr1++ & *ptr2++;
                }
                ptr1 += colSkip;
                ptr2 += colSkip;
                dst += colSkip;
            }
        }
    }

    // find pow2 rectangles
    std::vector<Rect> p2Rects;
    for (int rowDepth = 0; rowDepth <= log2[kernel.rows]; rowDepth++)
    {
        int rowSkip = (1 << rowDepth) - 1;
        int rowLim = kernel.rows - rowSkip;
        for (int colDepth = 0; colDepth <= log2[kernel.cols]; colDepth++)
        {
            int colSkip = (1 << colDepth) - 1;
            int colLim = kernel.cols - colSkip;

            uchar* ptr = st[rowDepth][colDepth].ptr();
            for (int row = 0; row < rowLim; row++)
            {
                for (int col = 0; col < colLim; col++, ptr++)
                {
                    // ignore black cell
                    if (ptr[0] == 0) continue;

                    // ignore if both sides are white by each axis
                    if (col > 0 && ptr[-1] == 1 && col < colLim && ptr[1] == 1) continue;
                    if (row > 0 && ptr[-kernel.cols] && row < rowLim && ptr[kernel.cols] == 1) continue;

                    // ignore one of neighbor block is white; will be alive in deeper table
                    if (col + (1 << colDepth) <= colLim && ptr[1 << colDepth] == 1) continue;
                    if (col - (1 << colDepth) >= 0 && ptr[-(1 << colDepth)] == 1) continue;
                    if (row + (1 << rowDepth) <= rowLim && ptr[(1 << rowDepth) * kernel.cols] == 1) continue;
                    if (row - (1 << rowDepth) >= 0 && ptr[-(1 << rowDepth) * kernel.cols] == 1) continue;

                    p2Rects.emplace_back(col, row, colDepth, rowDepth);
                }
                ptr += colSkip;
            }
        }
    }

    return p2Rects;
}

/*
*
*
* AtCoder: https://atcoder.jp/contests/ahc037/tasks/ahc037_a
*
* The rectilinear steiner arborescence problem
* https://link.springer.com/article/10.1007/BF01758762
*
*/
static std::vector<StStep> makePlan(std::vector<std::vector<bool>> sparseMatMap)
{
    auto comparePos = [](Point lp, Point rp) {
        int diffx = lp.x - rp.x;
        int diffy = lp.y - rp.y;
        int diff = diffx + diffy;
        if (diff != 0) return diff < 0;
        if (diffx != 0) return diffx < 0;
        return diffy < 0;
        };
    std::priority_queue<Point, std::vector<Point>, decltype(comparePos)> points{ comparePos };
    for (int r = 0; r < sparseMatMap.size(); r++)
    {
        for (int c = 0; c < sparseMatMap[r].size(); c++)
        {
            if (sparseMatMap[r][c]) points.push(Point(c, r));
        }
    }

    std::vector<StStep> ans;
    while (points.size() >= 2)
    {
        Point p1 = points.top();
        points.pop();
        Point p2 = points.top();
        points.pop();
        int newX = min(p1.x, p2.x);
        int newY = min(p1.y, p2.y);
        points.push(Point(newX, newY));

        for (int col = p1.x - 1; col >= newX; col--) ans.emplace_back(p1.y, col, Dim::Col);
        for (int row = p1.y - 1; row >= newY; row--) ans.emplace_back(row, p1.x, Dim::Row);
        for (int col = p2.x - 1; col >= newX; col--) ans.emplace_back(p2.y, col, Dim::Col);
        for (int row = p2.y - 1; row >= newY; row--) ans.emplace_back(row, p2.x, Dim::Row);
    }
    Point p1 = points.top();
    for (int col = p1.x - 1; col >= 0; col--) ans.emplace_back(p1.y, col, Dim::Col);
    for (int row = p1.y - 1; row >= 0; row--) ans.emplace_back(row, 0, Dim::Row);
    std::reverse(ans.begin(), ans.end());
    return ans;
}

static void makeMinStMat(InputArray src, OutputArray dst, int rowStep, int colStep)
{
    CV_Assert(rowStep * colStep == 0); // one of "rowStep" or "colStep" is required to be 0.

    Mat src_ = src.getMat();
    Mat dst_ = dst.getMat();
    int rowLim = src.rows() - rowStep;
    int colChLim = (src.cols() - colStep) * src.channels();
    int borderSkipStep = colStep * src.channels();

    uchar* srcPtr1 = src_.ptr<uchar>(0, 0);
    uchar* srcPtr2 = src_.ptr<uchar>(rowStep, colStep);
    uchar* dstPtr = dst_.ptr<uchar>(0, 0);
    for (int row = 0; row < rowLim; row++)
    {
        for (int colCh = 0; colCh < colChLim; colCh++)
        {
            if (*srcPtr1 < *srcPtr2)
            {
                *dstPtr++ = *srcPtr1++;
                srcPtr2++;
            }
            else
            {
                *dstPtr++ = *srcPtr2++;
                srcPtr1++;
            }
        }
        srcPtr1 += borderSkipStep;
        srcPtr2 += borderSkipStep;
        dstPtr += borderSkipStep;
    }
}
static void makeMaxStMat(InputArray src, OutputArray dst, int rowStep, int colStep)
{
    CV_Assert(rowStep * colStep == 0); // one of "rowStep" or "colStep" is required to be 0.

    Mat src_ = src.getMat();
    Mat dst_ = dst.getMat();
    int rowLim = src.rows() - rowStep;
    int colChLim = (src.cols() - colStep) * src.channels();
    int borderSkipStep = colStep * src.channels();

    uchar* srcPtr1 = src_.ptr<uchar>(0, 0);
    uchar* srcPtr2 = src_.ptr<uchar>(rowStep, colStep);
    uchar* dstPtr = dst_.ptr<uchar>(0, 0);
    for (int row = 0; row < rowLim; row++)
    {
        for (int colCh = 0; colCh < colChLim; colCh++)
        {
            if (*srcPtr1 > *srcPtr2)
            {
                *dstPtr++ = *srcPtr1++;
                srcPtr2++;
            }
            else
            {
                *dstPtr++ = *srcPtr2++;
                srcPtr1++;
            }
        }
        srcPtr1 += borderSkipStep;
        srcPtr2 += borderSkipStep;
        dstPtr += borderSkipStep;
    }
}

void dilate(InputArray src, OutputArray dst, InputArray kernel, Point anchor,
    int borderType, const Scalar& borderValue)
{
}

void erode(InputArray _src, OutputArray _dst, InputArray _kernel, Point anchor,
    int borderType, const Scalar& borderValue)
{
    //---------------------------
    // checking input
    uchar ZERO = 255;

    Mat src = _src.getMat();
    Mat dst = _dst.getMat();
    Mat kernel = _kernel.getMat();
    anchor = stMorph::normalizeAnchor(anchor, kernel.size());

    // iterations; is it needed yet?
    // borderType; BORDER_CONSTANT

    dst.setTo(ZERO);
    Scalar bV = borderValue;
    if (borderType == cv::BorderTypes::BORDER_CONSTANT && borderValue == cv::morphologyDefaultBorderValue())
    {
        bV = Scalar::all(ZERO);
        // see morph.dispatch.cpp:111
        // need to think CV_8U/CV_16U/CV_16S/CV_32F/CV64F
    }

    //---------------------------
    // pre processing

    // adding border to the source.
    // borderType := cv::BorderTypes::BORDER_CONSTANT(0)
    // borderValue := DBL_MAX => { MAX_VALUE (when erasion); MIN_VALUE (when dilation) }
    Mat expandedSrc(src.rows + kernel.rows, src.cols + kernel.cols, src.type());
    cv::copyMakeBorder(src, expandedSrc, anchor.y, kernel.cols - 1 - anchor.y, anchor.x, kernel.rows - 1 - anchor.x, borderType, bV);

    // log2 table construction
    int len = max(kernel.rows, kernel.cols) + 1;
    std::vector<int> log2(len);
    for (int i = 2; i < len; i++) log2[i] = log2[i >> 1] + 1;

    // カーネルから幅と高さが2のべき乗の長方形リストを生成する
    // - リストの width と height は log2 をとった値
    std::vector<Rect> pow2Rects = genPow2RectsToCoverKernel(kernel);

    // 幅と高さが2のべき乗の長方形リストを幅と高さごとに集計
    std::vector<std::vector<bool>> sparseMatMap(log2[kernel.rows] + 1, std::vector<bool>(log2[kernel.cols] + 1, false));
    for (int i = 0; i < pow2Rects.size(); i++) sparseMatMap[pow2Rects[i].height][pow2Rects[i].width] = true;

    // スパーステーブルの生成計画を立てる; planning how to calculate required mats in sparse table
    std::vector<StStep> stProcess = makePlan(sparseMatMap);

    // スパーステーブルの生成; generate sparse table
    std::vector<std::vector<Mat*>> st(log2[kernel.rows] + 1, std::vector<Mat*>(log2[kernel.cols] + 1));
    st[0][0] = &expandedSrc;
    for (int i = 0; i < stProcess.size(); i++)
    {
        StStep step = stProcess[i];
        switch (step.ax)
        {
        case Dim::Col:
            st[step.dimRow][step.dimCol + 1] = new Mat(expandedSrc.rows, expandedSrc.cols, expandedSrc.type());
            makeMinStMat(*st[step.dimRow][step.dimCol], *st[step.dimRow][step.dimCol + 1], 0, 1 << step.dimCol);
            break;
        case Dim::Row:
            st[step.dimRow + 1][step.dimCol] = new Mat(expandedSrc.rows, expandedSrc.cols, expandedSrc.type());
            makeMinStMat(*st[step.dimRow][step.dimCol], *st[step.dimRow + 1][step.dimCol], 1 << step.dimRow, 0);
            break;
        }
    }

    // 結果構築; construct the result
    for (int i = 0; i < pow2Rects.size(); i++)
    {
        Rect rect = pow2Rects[i];
        Mat sparseMat = *st[rect.height][rect.width];
        int sideBorderSkipStep = (kernel.cols - 1) * sparseMat.step.p[1];
        int colChLim = src.cols * src.channels();

        uchar* srcPtr = sparseMat.ptr(rect.y, rect.x);
        uchar* dstPtr = dst.ptr();
        for (int row = 0; row < src.rows; row++)
        {
            for (int col = 0; col < colChLim; col++)
            {
                if (*srcPtr < *dstPtr) *dstPtr = *srcPtr;
                srcPtr++;
                dstPtr++;
            }
            srcPtr += sideBorderSkipStep;
        }
    }
}

void morphologyEx(InputArray _src, OutputArray _dst, int op,
    InputArray _kernel, Point anchor, int iterations,
    int borderType, const Scalar& borderValue)
{
}

} // namespace st
} // namespace ximgproc
} // namespace cv
