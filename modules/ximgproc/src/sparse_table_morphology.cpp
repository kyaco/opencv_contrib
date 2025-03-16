// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include <limits>
#include <vector>
#include <algorithm>

namespace cv {
namespace stMorph {

// Generate list of rectangles whose width and height are power of 2.
// (The width and height values of returned rects ​​are the log2 of the actual values.)
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

template <typename T>
void morphOp(Op minmax, InputArray _src, OutputArray _dst, InputArray kernel,
    Point anchor, int iterations,
    int borderType, const Scalar& borderVal)
{
    T nil = (minmax == Op::Min) ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();
    std::vector<Rect> pow2Rects = genPow2RectsToCoverKernel(kernel);

    // get the depth limits;
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
    _dst.create(_src.size(), _src.type());
    Mat dst = _dst.getMat();

    // adding border to the source.
    Scalar bV = borderVal;
    if (borderType == BorderTypes::BORDER_CONSTANT && borderVal == morphologyDefaultBorderValue())
        bV = Scalar::all(nil);

    do
    {
        Mat expandedSrc;
        copyMakeBorder(src, expandedSrc,
            anchor.y, kernel.cols() - 1 - anchor.y,
            anchor.x, kernel.rows() - 1 - anchor.x,
            borderType, bV);

        dst.setTo(nil);

        // TODO: keep only needed memories.
        std::vector<std::vector<Mat>> st(rowDepthLim, std::vector<Mat>(colDepthLim));
        st[0][0] = expandedSrc;
        for (int i = 0; i < stPlan.size(); i++)
        {
            StStep step = stPlan[i];
            Mat& curr = st[step.dimRow][step.dimCol];
            Mat* dst1;
            int ofsX = 0, ofsY = 0;
            if (step.ax == Dim::Col)
            {
                ofsX = 1 << step.dimCol;
                dst1 = &st[step.dimRow][step.dimCol + 1];
            }
            else
            {
                ofsY = 1 << step.dimRow;
                dst1 = &st[step.dimRow + 1][step.dimCol];
            }
            int width = curr.cols - ofsX;
            int height = curr.rows - ofsY;
            Mat& src1 = st[step.dimRow][step.dimCol](Rect(0, 0, width, height));
            Mat& src2 = st[step.dimRow][step.dimCol](Rect(ofsX, ofsY, width, height));
            dst1->create(height, width, curr.type());
            if (minmax == Op::Min) cv::min(src1, src2, *dst1);
            else cv::max(src1, src2, *dst1);
        }

        // result constructioin
        for (int i = 0; i < pow2Rects.size(); i++)
        {
            Rect rect = pow2Rects[i];
            Rect srcRect = Rect(rect.x, rect.y, dst.cols, dst.rows);
            Mat& next = st[rect.height][rect.width](srcRect);
            if (minmax == Op::Min) cv::min(dst, next, dst);
            else cv::max(dst, next, dst);
        }

        src = dst;
    } while (--iterations > 0);
}

void morphOp(Op minmax, InputArray _src, OutputArray _dst, InputArray _kernel,
    Point anchor, int iterations,
    int borderType, const Scalar& borderVal)
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

    switch (_src.depth())
    {
    case CV_8U:
        morphOp<uchar>(minmax, _src, _dst, kernel, anchor, iterations, borderType, borderVal);
        return;
    case CV_8S:
        morphOp<char>(minmax, _src, _dst, kernel, anchor, iterations, borderType, borderVal);
        return;
    case CV_16U:
        morphOp<ushort>(minmax, _src, _dst, kernel, anchor, iterations, borderType, borderVal);
        return;
    case CV_16S:
        morphOp<short>(minmax, _src, _dst, kernel, anchor, iterations, borderType, borderVal);
        return;
    case CV_32S:
        morphOp<int>(minmax, _src, _dst, kernel, anchor, iterations, borderType, borderVal);
        return;
    case CV_32F:
        morphOp<float>(minmax, _src, _dst, kernel, anchor, iterations, borderType, borderVal);
        return;
    case CV_64F:
        morphOp<double>(minmax, _src, _dst, kernel, anchor, iterations, borderType, borderVal);
        return;
    }
}

void dilate(InputArray src, OutputArray dst, InputArray kernel,
    Point anchor, int iterations,
    int borderType, const Scalar& borderVal)
{
    morphOp(Op::Max, src, dst, kernel, anchor, iterations, borderType, borderVal);
}

void erode(InputArray src, OutputArray dst, InputArray kernel,
    Point anchor, int iterations,
    int borderType, const Scalar& borderVal)
{
    morphOp(Op::Min, src, dst, kernel, anchor, iterations, borderType, borderVal);
}

void morphologyEx(InputArray src, OutputArray dst, int op,
    InputArray kernel, Point anchor, int iterations,
    int borderType, const Scalar& borderVal)
{
    CV_INSTRUMENT_REGION();

    CV_Assert(!src.empty());

    Mat _kernel = kernel.getMat();
    if (_kernel.empty())
    {
        _kernel = getStructuringElement(MORPH_RECT, Size(3, 3), Point(1, 1));
    }

    Mat _src = src.getMat(), temp;
    dst.create(_src.size(), _src.type());
    Mat _dst = dst.getMat();

    switch (op)
    {
    case MORPH_ERODE:
        stMorph::erode(_src, _dst, _kernel, anchor, iterations, borderType, borderVal);
        break;
    case MORPH_DILATE:
        stMorph::dilate(_src, _dst, _kernel, anchor, iterations, borderType, borderVal);
        break;
    case MORPH_OPEN:
        stMorph::erode(_src, _dst, _kernel, anchor, iterations, borderType, borderVal);
        stMorph::dilate(_dst, _dst, _kernel, anchor, iterations, borderType, borderVal);
        break;
    case MORPH_CLOSE:
        stMorph::dilate(_src, _dst, _kernel, anchor, iterations, borderType, borderVal);
        stMorph::erode(_dst, _dst, _kernel, anchor, iterations, borderType, borderVal);
        break;
    case MORPH_GRADIENT:
        stMorph::erode(_src, temp, _kernel, anchor, iterations, borderType, borderVal);
        stMorph::dilate(_src, _dst, _kernel, anchor, iterations, borderType, borderVal);
        _dst -= temp;
        break;
    case MORPH_TOPHAT:
        if (_src.data != _dst.data)
            temp = _dst;
        stMorph::erode(_src, temp, _kernel, anchor, iterations, borderType, borderVal);
        stMorph::dilate(temp, temp, _kernel, anchor, iterations, borderType, borderVal);
        _dst = _src - temp;
        break;
    case MORPH_BLACKHAT:
        if (_src.data != _dst.data)
            temp = _dst;
        stMorph::dilate(_src, temp, _kernel, anchor, iterations, borderType, borderVal);
        stMorph::erode(temp, temp, _kernel, anchor, iterations, borderType, borderVal);
        _dst = temp - _src;
        break;
    case MORPH_HITMISS:
        CV_Assert(_src.type() == CV_8UC1);
        if (countNonZero(_kernel) <= 0)
        {
            _src.copyTo(_dst);
            break;
        }
        {
            Mat k1, k2, e1, e2;
            k1 = (_kernel == 1);
            k2 = (_kernel == -1);

            if (countNonZero(k1) <= 0)
                e1 = Mat(_src.size(), _src.type(), Scalar(255));
            else
                stMorph::erode(_src, e1, k1, anchor, iterations, borderType, borderVal);

            if (countNonZero(k2) <= 0)
                e2 = Mat(_src.size(), _src.type(), Scalar(255));
            else
            {
                Mat _src_complement;
                bitwise_not(_src, _src_complement);
                stMorph::erode(_src_complement, e2, k2, anchor, iterations, borderType, borderVal);
            }
            _dst = e1 & e2;
        }
        break;
    default:
        CV_Error(cv::Error::StsBadArg, "unknown morphological operation");
    }
}

}} // cv::st::
