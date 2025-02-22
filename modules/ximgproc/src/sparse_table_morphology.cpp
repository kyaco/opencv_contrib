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
    // op = ...?

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

    // log2 table construction
    int len = max(kernel.rows, kernel.cols) + 1;
    int* lg = new int[len];
    lg[1] = 0;
    for (int i = 2; i < len; i++) lg[i] = lg[i >> 1] + 1;

    // generating a set of rectangles that covers whole kernel
    // todo: implement good algorithm
    int kCount = 0;
    int buffSize = kernel.rows * 2;
    Rect* rects = new Rect[buffSize];
    if (rects != NULL)
    {
        // bad implementation; just separating by line.
        for (int row = 0; row < kernel.rows; row++)
        {
            uchar pre = 0;
            for (int col = 0; col < kernel.cols; col++)
            {
                if (kernel.ptr(row)[col] == 0)
                {
                    if (pre == 1)
                    {
                        rects[kCount].width = col - rects[kCount].x;
                        kCount++;
                    }
                }
                else
                {
                    if (pre == 0)
                    {
                        rects[kCount].y = row;
                        rects[kCount].height = 1;
                        rects[kCount].x = col;
                    }
                }
                pre = kernel.ptr(row)[col];
            }
            if (pre == 1)
            {
                rects[kCount].width = kernel.cols - rects[kCount].x;
                kCount++;
            }
        }
    }

    // calculate required mats in sparsetable
    //   sparseTable[lnKcol][lnKrow] can be calculated from
    //   sparseTable[lnKcol - 1][lnKrow] or sparseTable[lnKcol][lnKrow - 1].
    //
    // todo: implement better algorithm.
    Mat stRequiredMatMap(lg[kernel.rows] + 1, lg[kernel.cols] + 1, CV_8UC1);
    stRequiredMatMap.setTo(0);
    for (int i = 0; i < kCount; i++)
    {
        stRequiredMatMap.ptr(lg[rects[i].height])[lg[rects[i].width]] = 1;
    }
#if 0
    cv::resize(stRequiredMatMap, stRequiredMatMap, cv::Size(), 10, 10, 0);
    imshow("debug", stRequiredMatMap);
#endif

    // temporary implementation; only row separation is supported.
    int szColDepth = stRequiredMatMap.cols;
    int szRowDepth = 1;

    int stSizes[] = { szRowDepth, szColDepth, expandedSrc.rows, expandedSrc.cols };
    Mat sparseTable(4, stSizes, src.type(), Scalar(0));

    // sparse table construction
    uchar* ptr = sparseTable.ptr();
    uchar* refPtr = expandedSrc.ptr();

    for (int row = 0; row < expandedSrc.rows; row++)
    {
        for (int col = 0; col < expandedSrc.cols; col++)
        {
            for (unsigned int c = 0; c < src.channels(); c++)
            {
                *ptr = *refPtr;
                ptr++;
                refPtr++;
            }
        }
    }
    for (int lgColCnt = 1; lgColCnt < szColDepth; lgColCnt++)
    {
        int b = (1 << lgColCnt) - 1;
        int colROfs = sparseTable.step.p[3] * (1 << (lgColCnt - 1));
        int colSkipOfs = b * sparseTable.step.p[3];

        for (int row = 0; row < expandedSrc.rows; row++)
        {
            for (int col = 0; col < expandedSrc.cols - b; col++)
            {
                for (unsigned int c = 0; c < src.channels(); c++)
                {
                    uchar* l = ptr - sparseTable.step.p[1];
                    uchar* r = l + colROfs;
                    *ptr = min(*l, *r);
                    ptr++;
                }
            }
            ptr += colSkipOfs;
        }
    }
    for (int lgRowCnt = 1; lgRowCnt < szRowDepth; lgRowCnt++)
    {
        int a = (1 << lgRowCnt) - 1;
        int rowROfs = sparseTable.step.p[2] * (1 << (lgRowCnt - 1));
        for (int lgColCnt = 0; lgColCnt < szColDepth; lgColCnt++)
        {
            int b = (1 << lgColCnt) - 1;
            int colROfs = sparseTable.step.p[3] * (1 << (lgColCnt - 1));
            int colSkipOfs = b * sparseTable.step.p[3];

            for (int row = 0; row < expandedSrc.rows - a; row++)
            {
                for (int col = 0; col < expandedSrc.cols - b; col++)
                {
                    for (unsigned int c = 0; c < src.channels(); c++)
                    {
                        uchar* l = ptr - sparseTable.step.p[0];
                        uchar* r = l + rowROfs;
                        *ptr = min(*l, *r);
                        ptr++;
                    }
                }
                ptr += colSkipOfs;
            }
            ptr += a * sparseTable.step.p[2];
        }
    }

    // result construction
    for (int i = 0; i < kCount; i++)
    {
        int lgRectRows = lg[rects[i].height];
        int lgRectCols = lg[rects[i].width];
        int ofsTB = (rects[i].height - (1 << lgRectRows)) * sparseTable.step.p[2];
        int ofsLR = (rects[i].width - (1 << lgRectCols)) * sparseTable.step.p[3];
        int sideBorderSkipStep = (kernel.cols - 1) * sparseTable.step.p[3];
        uchar* vLT = sparseTable.ptr()
                    + sparseTable.step.p[0] * lgRectRows
                    + sparseTable.step.p[1] * lgRectCols
                    + rects[i].y * sparseTable.step.p[2]
                    + rects[i].x * sparseTable.step.p[3];
        uchar* vLB = vLT + ofsTB;
        uchar* vRT = vLT + ofsLR;
        uchar* vRB = vRT + ofsTB;
        uchar* dstPtr = dst.ptr();

        for (int row = 0; row < src.rows; row++)
        {
            for (int col = 0; col < src.cols; col++)
            {
                for (int c = 0; c < src.channels(); c++)
                {
                    *dstPtr = min(*dstPtr, min(min(*vLT, *vLB), min(*vRT, *vRB)));
                    vLT++;
                    vLB++;
                    vRT++;
                    vRB++;
                    dstPtr++;
                }
            }
            vLT += sideBorderSkipStep;
            vLB += sideBorderSkipStep;
            vRT += sideBorderSkipStep;
            vRB += sideBorderSkipStep;
        }
    }
    delete[] lg;
    delete[] rects;
}

void morphologyEx(InputArray _src, OutputArray _dst, int op,
    InputArray _kernel, Point anchor, int iterations,
    int borderType, const Scalar& borderValue)
{
}

} // namespace st
} // namespace ximgproc
} // namespace cv
