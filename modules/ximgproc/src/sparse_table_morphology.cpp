/*
 *  By downloading, copying, installing or using the software you agree to this license.
 *  If you do not agree to this license, do not download, install,
 *  copy or use the software.
 *
 *
 *  License Agreement
 *  For Open Source Computer Vision Library
 *  (3 - clause BSD License)
 *
 *  Redistribution and use in source and binary forms, with or without modification,
 *  are permitted provided that the following conditions are met :
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *  this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *  this list of conditions and the following disclaimer in the documentation
 *  and / or other materials provided with the distribution.
 *
 *  * Neither the names of the copyright holders nor the names of the contributors
 *  may be used to endorse or promote products derived from this software
 *  without specific prior written permission.
 *
 *  This software is provided by the copyright holders and contributors "as is" and
 *  any express or implied warranties, including, but not limited to, the implied
 *  warranties of merchantability and fitness for a particular purpose are disclaimed.
 *  In no event shall copyright holders or contributors be liable for any direct,
 *  indirect, incidental, special, exemplary, or consequential damages
 *  (including, but not limited to, procurement of substitute goods or services;
 *  loss of use, data, or profits; or business interruption) however caused
 *  and on any theory of liability, whether in contract, strict liability,
 *  or tort(including negligence or otherwise) arising in any way out of
 *  the use of this software, even if advised of the possibility of such damage.
 */
#include "precomp.hpp"
#include <math.h>
#include <vector>
#include <iostream>
#include<stack>
#include<algorithm>

namespace cv {
namespace ximgproc {
namespace st {

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

static std::vector<Rect> GetCoveringRectangles(InputArray _kernel)
{
    std::vector<Rect> rects;
    Mat kernel = _kernel.getMat();
    int kCount = 0;

    // 行ごとで四角を作るだけの実装
    for (int row = 0; row < kernel.rows; row++)
    {
        uchar pre = 0;
        for (int col = 0; col < kernel.cols; col++)
        {
            if (pre == 1 && kernel.ptr(row)[col] == 0)
            {
                rects[kCount].width = col - rects[kCount].x;
                kCount++;
            }
            if (pre == 0 && kernel.ptr(row)[col] == 1)
            {
                rects.emplace_back(col, row, 0, 1);
            }
            pre = kernel.ptr(row)[col];
        }
        if (pre == 1)
        {
            rects[kCount].width = kernel.cols - rects[kCount].x;
            kCount++;
        }
    }
    return rects;
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

std::vector<StStep> makePlan(std::vector<std::vector<bool>> sparseMatMap)
{
    std::vector<StStep> ans;
    std::vector<std::vector<bool>> visitedMap(sparseMatMap.size(), std::vector<bool>(sparseMatMap[0].size(), false));
    visitedMap[0][0] = true;
    for (int row = 0; row < sparseMatMap.size(); row++)
    {
        for (int col = 0; col < sparseMatMap[row].size(); col++)
        {
            if (sparseMatMap[row][col])
            {
                for (int c = 0; c <= col; c++)
                {
                    if (!visitedMap[0][c])
                    {
                        visitedMap[0][c] = true;
                        ans.emplace_back(0, c - 1, Dim::Col);
                    }
                }
                for (int r = 0; r <= row; r++)
                {
                    if (!visitedMap[r][col])
                    {
                        visitedMap[r][col] = true;
                        ans.emplace_back(r - 1, col, Dim::Row);
                    }
                }
            }
        }
    }
    return ans;
}

void MakeMinStMat(InputArray src, OutputArray dst, int rowStep, int colStep)
{
    CV_Assert(rowStep * colStep == 0); // one of "rowStep" or "colStep" is required to be 0.

    Mat src_ = src.getMat();
    Mat dst_ = dst.getMat();
    uchar* srcPtr1 = src_.ptr<uchar>(0, 0);
    uchar* srcPtr2 = src_.ptr<uchar>(rowStep, colStep);
    uchar* dstPtr = dst_.ptr<uchar>(0, 0);
    for (int row = 0; row < src.rows() - rowStep; row++)
    {
        for (int col = 0; col < src.cols() - colStep; col++)
        {
            for (int c = 0; c < src.channels(); c++)
            {
                *dstPtr = min(*srcPtr1, *srcPtr2);
                srcPtr1++;
                srcPtr2++;
                dstPtr++;
            }
        }
        srcPtr1 += colStep * src.channels();
        srcPtr2 += colStep * src.channels();
        dstPtr += colStep * src.channels();
    }
}
void MakeMaxStMat(InputArray src, OutputArray dst, int rowStep, int colStep)
{
    CV_Assert(rowStep * colStep == 0); // one of "rowStep" or "colStep" is required to be 0.

    Mat src_ = src.getMat();
    Mat dst_ = dst.getMat();
    uchar* srcPtr1 = src_.ptr<uchar>(0, 0);
    uchar* srcPtr2 = src_.ptr<uchar>(rowStep, colStep);
    uchar* dstPtr = dst_.ptr<uchar>(0, 0);
    for (int row = 0; row < src.rows(); row++)
    {
        for (int col = 0; col < src.cols(); col++)
        {
            *dstPtr = max(*srcPtr1, *srcPtr2);
        }
    }
}

void dilate(InputArray src, OutputArray dst, InputArray kernel,
    Point anchor, int iterations,
    int borderType, const Scalar& borderValue)
{
}

void erode(InputArray _src, OutputArray _dst, InputArray _kernel,
    Point anchor, int iterations,
    int borderType, const Scalar& borderValue)
{
    //---------------------------
    // checking input
    uchar ZERO = 255;

    Mat src = _src.getMat();
    Mat dst = _dst.getMat();
    Mat kernel = _kernel.getMat();
    anchor = st::normalizeAnchor(anchor, kernel.size());

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

    // generating a set of rectangles that covers whole kernel
    std::vector<Rect> rects = GetCoveringRectangles(kernel);

    // log2 table construction
    int len = max(kernel.rows, kernel.cols) + 1;
    std::vector<int> lg(len);
    for (int i = 2; i < len; i++) lg[i] = lg[i >> 1] + 1;

    // 矩形を2冪矩形に分解 & 登場した2冪矩形の情報を位置と幅高さの指数で記録
    std::vector<std::vector<bool>> sparseMatMap(lg[kernel.rows] + 1, std::vector<bool>(lg[kernel.cols] + 1, false));
    std::vector<Rect> powerOf2Rects;
    for (int i = 0; i < rects.size(); i++)
    {
        Rect rect = rects[i];
        int lgCols = lg[rect.width];
        int lgRows = lg[rect.height];
        bool isColDivisionRequired = (1 << lgCols) < rect.width;
        bool isRowDivisionRequired = (1 << lgRows) < rect.height;

        sparseMatMap[lgRows][lgCols] = true;

        powerOf2Rects.emplace_back(rect.x, rect.y, lgCols, lgRows);
        if (isColDivisionRequired)
            powerOf2Rects.emplace_back(rect.x + rect.width - (1 << lgCols), rect.y, lgCols, lgRows);
        if (isRowDivisionRequired)
            powerOf2Rects.emplace_back(rect.x + rect.width, rect.y - (1 << lgRows), lgCols, lgRows);
        if (isColDivisionRequired && isRowDivisionRequired)
            powerOf2Rects.emplace_back(rect.x + rect.width - (1 << lgCols), rect.y - (1 << lgRows), lgCols, lgRows);
    }

    // スパーステーブルの生成計画を立てる; planning how to calculate required mats in sparsetable
    std::vector<StStep> stProcess = makePlan(sparseMatMap);

    // スパーステーブルの生成
    std::vector<std::vector<Mat*>> st = std::vector<std::vector<Mat*>>(lg[kernel.rows] + 1, std::vector<Mat*>(lg[kernel.cols] + 1));
    st[0][0] = &expandedSrc;
    for (int i = 0; i < stProcess.size(); i++)
    {
        StStep step = stProcess[i];
        switch (step.ax)
        {
        case Dim::Col:
            st[step.dimRow][step.dimCol + 1] = new Mat(expandedSrc.rows, expandedSrc.cols, expandedSrc.type());
            MakeMinStMat(*st[step.dimRow][step.dimCol], *st[step.dimRow][step.dimCol + 1], 0, 1 << step.dimCol);
            break;
        case Dim::Row:
            st[step.dimRow + 1][step.dimCol] = new Mat(expandedSrc.rows, expandedSrc.cols, expandedSrc.type());
            MakeMinStMat(*st[step.dimRow][step.dimCol], *st[step.dimRow + 1][step.dimCol], 1 << step.dimRow, 0);
            break;
        }
    }

    // 結果構築
    int aaa; //???
    for (int i = 0; i < powerOf2Rects.size(); i++)
    {
        Rect rect = powerOf2Rects[i];
        Mat* sparseMat = st[rect.height][rect.width];
        uchar* srcPtr = sparseMat->ptr() + sparseMat->step.p[0] * rect.y + sparseMat->step.p[1] * rect.x;
        uchar* dstPtr = dst.ptr();
        int sideBorderSkipStep = (kernel.cols - 1) * sparseMat->step.p[1];

        for (int row = 0; row < src.rows; row++)
        {
            for (int col = 0; col < src.cols; col++)
            {
                for (int c = 0; c < src.channels(); c++)
                {
                    *dstPtr = min(*dstPtr, *srcPtr);
                    srcPtr++;
                    dstPtr++;
                }
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
