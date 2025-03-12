// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include <limits>
#include <utility>
#include <vector>

namespace cv {
namespace stMorph {

std::vector<Rect> genPow2RectsToCoverKernel(InputArray _kernel)
{
    CV_Assert(_kernel.type() == CV_8UC1);

    Mat kernel = _kernel.getMat();

    // generate log2 table
    int len = std::max(kernel.rows, kernel.cols) + 1;
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

        st[0][colDepth].create(kernel.size(), kernel.type());
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

            st[rowDepth][colDepth].create(kernel.size(), kernel.type());
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
        int rowOfst = 1 << rowDepth;
        int rowSkip = rowOfst - 1;
        int rowLim = kernel.rows - rowSkip;
        int x = rowOfst * kernel.cols;
        for (int colDepth = 0; colDepth <= log2[kernel.cols]; colDepth++)
        {
            int colOfst = 1 << colDepth;
            int colSkip = colOfst - 1;
            int colLim = kernel.cols - colSkip;

            uchar* ptr = st[rowDepth][colDepth].ptr();
            for (int row = 0; row < rowLim; row++)
            {
                for (int col = 0; col < colLim; col++, ptr++)
                {
                    // ignore black cell
                    if (ptr[0] == 0) continue;

                    // ignore if both sides are white by each axis
                    if (col > 0 && ptr[-1] == 1
                        && col < colLim && ptr[1] == 1) continue;
                    if (row > 0 && ptr[-kernel.cols]
                        && row < rowLim && ptr[kernel.cols] == 1) continue;

                    // ignore one of neighbor block is white; will be alive in deeper table
                    if (col + colOfst <= colLim && ptr[colOfst] == 1) continue;
                    if (col - colOfst >= 0 && ptr[-colOfst] == 1) continue;
                    if (row + rowOfst <= rowLim && ptr[x] == 1) continue;
                    if (row - rowOfst >= 0 && ptr[-x] == 1) continue;

                    p2Rects.emplace_back(col, row, colDepth, rowDepth);
                }
                ptr += colSkip;
            }
        }
    }

    return p2Rects;
}

std::vector<StStep> planSparseTableConstr(std::vector<std::vector<bool>> sparseMatMap)
{
/*
*
*
* AtCoder: https://atcoder.jp/contests/ahc037/tasks/ahc037_a
*
* The rectilinear steiner arborescence problem
* https://link.springer.com/article/10.1007/BF01758762
*
*/
    std::vector<Point> pos;
    sparseMatMap[0][0] = true;
    for (int r = 0; r < sparseMatMap.size(); r++)
        for (int c = 0; c < sparseMatMap[r].size(); c++)
            if (sparseMatMap[r][c]) pos.emplace_back(c, r);
    std::vector<StStep> plan;
    while(pos.size() > 1)
    {
        int maxCost = -1;
        int maxI = 0;
        int maxJ = 0;
        int maxX = 0;
        int maxY = 0;
        for (int i = 0; i < pos.size(); i++)
        {
            for (int j = i + 1; j < pos.size(); j++)
            {
                int _x = std::min(pos[i].x, pos[j].x);
                int _y = std::min(pos[i].y, pos[j].y);
                int cost = _x + _y;
                if (maxCost < cost)
                {
                    maxCost = cost;
                    maxI = i;
                    maxJ = j;
                    maxX = _x;
                    maxY = _y;
                }
            }
        }
        for (int col = pos[maxI].x - 1; col >= maxX; col--)
            plan.emplace_back(pos[maxI].y, col, Dim::Col);
        for (int row = pos[maxI].y - 1; row >= maxY; row--)
            plan.emplace_back(row, maxX, Dim::Row);
        for (int col = pos[maxJ].x - 1; col >= maxX; col--)
            plan.emplace_back(pos[maxJ].y, col, Dim::Col);
        for (int row = pos[maxJ].y - 1; row >= maxY; row--)
            plan.emplace_back(row, maxX, Dim::Row);

        pos[maxI] = Point(maxX, maxY);
        swap(pos[maxJ], pos[pos.size() - 1]);
        pos.pop_back();
    }

    reverse(plan.begin(), plan.end());
    return plan;
}

#pragma region Dilation

template <typename T>
void makeMaxSparseTableMat(InputArray src, OutputArray dst, int rowStep, int colStep)
{
    CV_Assert(rowStep * colStep == 0); // one of "rowStep" or "colStep" is required to be 0.

    dst.create(src.size(), src.type());
    Mat src_ = src.getMat();
    Mat dst_ = dst.getMat();
    int rowLim = src.rows() - rowStep;
    int colChLim = (src.cols() - colStep) * src.channels();
    int borderSkipStep = colStep * src.channels();

    T* srcPtr1 = src_.ptr<T>(0, 0);
    T* srcPtr2 = src_.ptr<T>(rowStep, colStep);
    T* dstPtr = dst_.ptr<T>(0, 0);
    for (int row = 0; row < rowLim; row++)
    {
        for (int colCh = 0; colCh < colChLim; colCh++)
        {
            // Somehow std::max(a,b) or a>b?a:b are slower.
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

template <typename T>
void _dilate(InputArray _src, OutputArray _dst, InputArray _kernel,
    Point anchor, int iterations,
    int borderType, const Scalar& borderValue)
{
    Mat kernel = _kernel.getMat();

    // Generate list of rectangles whose width and height are power of 2.
    // (The width and height values of returned rects ​​are the log2 of the actual values.)
    std::vector<Rect> pow2Rects = genPow2RectsToCoverKernel(kernel);

    // get the depth limits
    int rowDepthLim = 0, colDepthLim = 0;
    for (int i = 0; i < pow2Rects.size(); i++)
    {
        if (rowDepthLim < pow2Rects[i].height) rowDepthLim = pow2Rects[i].height;
        if (colDepthLim < pow2Rects[i].width) colDepthLim = pow2Rects[i].width;
    }
    rowDepthLim++;
    colDepthLim++;

    // list up required sparse table nodes.
    std::vector<std::vector<bool>> sparseMatMap(rowDepthLim, std::vector<bool>(colDepthLim, false));
    for (int i = 0; i < pow2Rects.size(); i++)
        sparseMatMap[pow2Rects[i].height][pow2Rects[i].width] = true;

    // plan how to calculate required nodes of 2D sparse table.
    std::vector<StStep> stPlan = planSparseTableConstr(sparseMatMap);

    Mat src = _src.getMat();

    do
    {
        // adding border to the source.
        Scalar bV = borderValue;
        if (borderType == BorderTypes::BORDER_CONSTANT
            && borderValue == morphologyDefaultBorderValue())
            bV = Scalar::all(0);
        Mat expandedSrc(src.rows + kernel.rows, src.cols + kernel.cols, src.type());
        copyMakeBorder(src, expandedSrc,
            anchor.y, kernel.cols - 1 - anchor.y,
            anchor.x, kernel.rows - 1 - anchor.x,
            borderType, bV);

        _dst.create(_src.size(), _src.type());
        Mat dst = _dst.getMat();

        std::vector<std::vector<Mat>> st(rowDepthLim, std::vector<Mat>(colDepthLim));
        st[0][0] = expandedSrc;
        for (int i = 0; i < stPlan.size(); i++)
        {
            StStep step = stPlan[i];
            switch (step.ax)
            {
            case Dim::Col:
                makeMaxSparseTableMat<T>(st[step.dimRow][step.dimCol], st[step.dimRow][step.dimCol + 1],
                    0, 1 << step.dimCol);
                break;
            case Dim::Row:
                makeMaxSparseTableMat<T>(st[step.dimRow][step.dimCol], st[step.dimRow + 1][step.dimCol],
                    1 << step.dimRow, 0);
                break;
            }
        }

        // result constructioin
        dst.setTo(0);
        int colChLim = dst.cols * dst.channels();
        for (int i = 0; i < pow2Rects.size(); i++)
        {
            Rect rect = pow2Rects[i];
            Mat sparseMat = st[rect.height][rect.width];
            int sideBorderSkipStep = (kernel.cols - 1) * sparseMat.channels();
            T* srcPtr = sparseMat.ptr<T>(rect.y, rect.x);
            T* dstPtr = dst.ptr<T>();
            for (int row = 0; row < dst.rows; row++)
            {
                for (int col = 0; col < colChLim; col++)
                {
                    if (*srcPtr > *dstPtr) *dstPtr = *srcPtr;
                    srcPtr++;
                    dstPtr++;
                }
                srcPtr += sideBorderSkipStep;
            }
        }

        src = dst;
    } while (--iterations > 0);
}

void dilate(InputArray _src, OutputArray _dst, InputArray _kernel,
    Point anchor, int iterations,
    int borderType, const Scalar& borderValue)
{
    Mat kernel = _kernel.getMat();
    if (iterations == 0 || kernel.rows * kernel.cols == 1)
    {
        _src.copyTo(_dst);
        return;
    }
    // Fix kernel in case of it is empty.
    if (kernel.empty())
    {
        kernel = getStructuringElement(MORPH_RECT, Size(1 + iterations * 2, 1 + iterations * 2));
        anchor = Point(iterations, iterations);
        iterations = 1;
    }
    if (countNonZero(kernel) == 0)
    {
        kernel.at<uchar>(0, 0) = 1;
    }
    // Fix anchor to the center of the kernel.
    anchor = stMorph::normalizeAnchor(anchor, kernel.size());

    // dilate operation
    switch (_src.depth())
    {
    case CV_8U:
        _dilate<uchar>(_src, _dst, kernel, anchor, iterations, borderType, borderValue);
        return;
    case CV_8S:
        _dilate<char>(_src, _dst, kernel, anchor, iterations, borderType, borderValue);
        return;
    case CV_16U:
        _dilate<ushort>(_src, _dst, kernel, anchor, iterations, borderType, borderValue);
        return;
    case CV_16S:
        _dilate<short>(_src, _dst, kernel, anchor, iterations, borderType, borderValue);
        return;
    case CV_32S:
        _dilate<int>(_src, _dst, kernel, anchor, iterations, borderType, borderValue);
        return;
    case CV_32F:
        _dilate<float>(_src, _dst, kernel, anchor, iterations, borderType, borderValue);
        return;
    case CV_64F:
        _dilate<double>(_src, _dst, kernel, anchor, iterations, borderType, borderValue);
        return;
    }
}

#pragma endregion

#pragma region Erosion

template <typename T>
void makeMinSparseTableMat(InputArray src, OutputArray dst, int rowStep, int colStep)
{
    CV_Assert(rowStep * colStep == 0); // one of "rowStep" or "colStep" is required to be 0.

    dst.create(src.size(), src.type());
    Mat src_ = src.getMat();
    Mat dst_ = dst.getMat();
    int rowLim = src.rows() - rowStep;
    int colChLim = (src.cols() - colStep) * src.channels();
    int borderSkipStep = colStep * src.channels();

    T* srcPtr1 = src_.ptr<T>(0, 0);
    T* srcPtr2 = src_.ptr<T>(rowStep, colStep);
    T* dstPtr = dst_.ptr<T>(0, 0);
    for (int row = 0; row < rowLim; row++)
    {
        for (int colCh = 0; colCh < colChLim; colCh++)
        {
            // Somehow std::min(a,b) or a<b?a:b are slower.
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

template <typename T>
void _erode(InputArray _src, OutputArray _dst, InputArray _kernel,
    Point anchor, int iterations,
    int borderType, const Scalar& borderValue)
{
    Mat kernel = _kernel.getMat();

    // Generate list of rectangles whose width and height are power of 2.
    // (The width and height values of returned rects ​​are the log2 of the actual values.)
    std::vector<Rect> pow2Rects = genPow2RectsToCoverKernel(kernel);

    // get the depth limits
    int rowDepthLim = 0, colDepthLim = 0;
    for (int i = 0; i < pow2Rects.size(); i++)
    {
        if (rowDepthLim < pow2Rects[i].height) rowDepthLim = pow2Rects[i].height;
        if (colDepthLim < pow2Rects[i].width) colDepthLim = pow2Rects[i].width;
    }
    rowDepthLim++;
    colDepthLim++;

    // list up required sparse table nodes.
    std::vector<std::vector<bool>> sparseMatMap(rowDepthLim, std::vector<bool>(colDepthLim, false));
    for (int i = 0; i < pow2Rects.size(); i++)
        sparseMatMap[pow2Rects[i].height][pow2Rects[i].width] = true;

    // plan how to calculate required nodes of 2D sparse table.
    std::vector<StStep> stPlan = planSparseTableConstr(sparseMatMap);

    Mat src = _src.getMat();

    do
    {
        // adding border to the source.
        Scalar bV = borderValue;
        if (borderType == BorderTypes::BORDER_CONSTANT
            && borderValue == morphologyDefaultBorderValue())
            bV = Scalar::all(std::numeric_limits<T>::max());
        Mat expandedSrc(src.rows + kernel.rows, src.cols + kernel.cols, src.type());
        copyMakeBorder(src, expandedSrc,
            anchor.y, kernel.cols - 1 - anchor.y,
            anchor.x, kernel.rows - 1 - anchor.x,
            borderType, bV);

        _dst.create(_src.size(), _src.type());
        Mat dst = _dst.getMat();

        std::vector<std::vector<Mat>> st(rowDepthLim, std::vector<Mat>(colDepthLim));
        st[0][0] = expandedSrc;
        for (int i = 0; i < stPlan.size(); i++)
        {
            StStep step = stPlan[i];
            switch (step.ax)
            {
            case Dim::Col:
                makeMinSparseTableMat<T>(st[step.dimRow][step.dimCol], st[step.dimRow][step.dimCol + 1],
                    0, 1 << step.dimCol);
                break;
            case Dim::Row:
                makeMinSparseTableMat<T>(st[step.dimRow][step.dimCol], st[step.dimRow + 1][step.dimCol],
                    1 << step.dimRow, 0);
                break;
            }
        }

        // result constructioin
        dst.setTo(std::numeric_limits<T>::max());
        int colChLim = dst.cols * dst.channels();
        for (int i = 0; i < pow2Rects.size(); i++)
        {
            Rect rect = pow2Rects[i];
            Mat sparseMat = st[rect.height][rect.width];
            int sideBorderSkipStep = (kernel.cols - 1) * sparseMat.channels();
            T* srcPtr = sparseMat.ptr<T>(rect.y, rect.x);
            T* dstPtr = dst.ptr<T>();
            for (int row = 0; row < dst.rows; row++)
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

        src = dst;
    } while (--iterations > 0);
}

void erode(InputArray _src, OutputArray _dst, InputArray _kernel,
    Point anchor, int iterations,
    int borderType, const Scalar& borderValue)
{
    Mat kernel = _kernel.getMat();
    if (iterations == 0 || kernel.rows * kernel.cols == 1)
    {
        _src.copyTo(_dst);
        return;
    }
    // Fix kernel in case of it is empty.
    if (kernel.empty())
    {
        kernel = getStructuringElement(MORPH_RECT, Size(1 + iterations * 2, 1 + iterations * 2));
        anchor = Point(iterations, iterations);
        iterations = 1;
    }
    if (countNonZero(kernel) == 0)
    {
        kernel.at<uchar>(0, 0) = 1;
    }
    // Fix anchor to the center of the kernel.
    anchor = stMorph::normalizeAnchor(anchor, kernel.size());

    // erode operation
    switch (_src.depth())
    {
    case CV_8U:
        _erode<uchar>(_src, _dst, kernel, anchor, iterations, borderType, borderValue);
        return;
    case CV_8S:
        _erode<char>(_src, _dst, kernel, anchor, iterations, borderType, borderValue);
        return;
    case CV_16U:
        _erode<ushort>(_src, _dst, kernel, anchor, iterations, borderType, borderValue);
        return;
    case CV_16S:
        _erode<short>(_src, _dst, kernel, anchor, iterations, borderType, borderValue);
        return;
    case CV_32S:
        _erode<int>(_src, _dst, kernel, anchor, iterations, borderType, borderValue);
        return;
    case CV_32F:
        _erode<float>(_src, _dst, kernel, anchor, iterations, borderType, borderValue);
        return;
    case CV_64F:
        _erode<double>(_src, _dst, kernel, anchor, iterations, borderType, borderValue);
        return;
    }
}

#pragma endregion

void morphologyEx(InputArray _src, OutputArray _dst, int op,
    InputArray _kernel, Point anchor, int iterations,
    int borderType, const Scalar& borderValue)
{
}

}} // cv::st::
