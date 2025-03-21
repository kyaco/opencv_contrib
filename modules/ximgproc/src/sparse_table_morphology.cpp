// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include <limits>
#include <vector>
#include <utility>

namespace cv {
namespace stMorph {

int log2(int n)
{
    int ans = -1;
    while (n > 0)
    {
        n /= 2;
        ans++;
    }
    return ans;
}

int longestRowRunLength(const Mat& kernel)
{
    int cnt = 0;
    int maxLen = 0;
    for (int c = 0; c < kernel.cols; c++)
    {
        cnt = 0;
        for (int r = 0; r < kernel.rows; r++)
        {
            if (kernel.at<uchar>(r, c) == 0)
            {
                maxLen = std::max(maxLen, cnt);
                cnt = 0;
            }
            else cnt++;
        }
        maxLen = std::max(maxLen, cnt);
    }
    return maxLen;
}

int longestColRunLength(const Mat& kernel)
{
    int cnt = 0;
    int maxLen = 0;
    for (int r = 0; r < kernel.rows; r++)
    {
        cnt = 0;
        for (int c = 0; c < kernel.cols; c++)
        {
            if (kernel.at<uchar>(r, c) == 0)
            {
                maxLen = std::max(maxLen, cnt);
                cnt = 0;
            }
            else cnt++;
        }
        maxLen = std::max(maxLen, cnt);
    }
    return maxLen;
}

std::vector<Point> findSeeds(const Mat& stNode, int rowDepth, int colDepth)
{
    int rowOfst = 1 << rowDepth;
    int colOfst = 1 << colDepth;
    std::vector<Point> p2Rects;
    for (int row = 0; row < stNode.rows; row++)
    {
        for (int col = 0; col < stNode.cols; col++)
        {
            // select white cells
            if (stNode.at<uchar>(row, col) == 0) continue;

            // select corner cells
            if (col > 0 && stNode.at<uchar>(row, col - 1) == 1
                && col + 1 < stNode.cols && stNode.at<uchar>(row, col + 1) == 1) continue;
            if (row > 0 && stNode.at<uchar>(row - 1, col) == 1
                && row + 1 < stNode.rows && stNode.at<uchar>(row + 1, col) == 1) continue;

            // zignore if neighboring block is white; will be alive in deeper table
            if (col + colOfst < stNode.cols && stNode.at<uchar>(row, col + colOfst) == 1) continue;
            if (col - colOfst >= 0 && stNode.at<uchar>(row, col - colOfst) == 1) continue;
            if (row + rowOfst < stNode.rows && stNode.at<uchar>(row + rowOfst, col) == 1) continue;
            if (row - rowOfst >= 0 && stNode.at<uchar>(row - rowOfst, col) == 1) continue;

            p2Rects.emplace_back(col, row);
        }
    }
    return p2Rects;
}

// Generate list of rectangles whose width and height are power of 2.
// (The width and height values of returned rects ​​are the log2 of the actual values.)
std::vector<std::vector<std::vector<Point>>> genPow2RectsToCoverKernel(
    const Mat& kernel, int rowDepthLim, int colDepthLim)
{
    CV_Assert(kernel.type() == CV_8UC1);

    std::vector<std::vector<std::vector<Point>>> p2Rects;
    Mat stCache = kernel;
    for (int rowDepth = 0; rowDepth < rowDepthLim; rowDepth++)
    {
        Mat st = stCache.clone();
        p2Rects.emplace_back(std::vector<std::vector<Point>>());
        for (int colDepth = 0; colDepth < colDepthLim; colDepth++)
        {
            p2Rects[rowDepth].emplace_back(findSeeds(st, rowDepth, colDepth));
            int colStep = 1 << colDepth;
            if (st.cols - colStep < 0) break;
            Rect s1(0, 0, st.cols - colStep, st.rows);
            Rect s2(colStep, 0, st.cols - colStep, st.rows);
            cv::min(st(s1), st(s2), st);
        }
        int rowStep = 1 << rowDepth;
        if (stCache.rows - rowStep < 0) break;
        Rect s1(0, 0, stCache.cols, stCache.rows - rowStep);
        Rect s2(0, rowStep, stCache.cols, stCache.rows - rowStep);
        cv::min(stCache(s1), stCache(s2), stCache);
    }

    // todo: implement greedy algorithm to minimize the rectangle set covering the kernel.

    return p2Rects;
}

std::vector<StStep> planSparseTableConstr(
    std::vector<std::vector<std::vector<Point>>> pow2Rects, int rowDepthLim, int colDepthLim,
    StStrategy strategy)
{
    // list up required sparse table nodes.
    std::vector<std::vector<bool>> sparseMatMap(rowDepthLim, std::vector<bool>(colDepthLim, false));
    for (int r = 0; r < pow2Rects.size(); r++)
    {
        for (int c = 0; c < pow2Rects[r].size(); c++)
        {
            if (pow2Rects[r][c].size() > 0) sparseMatMap[r][c] = true;
        }
    }
    sparseMatMap[0][0] = true;

    switch (strategy)
    {
    case Faster:
    {
        /*
        *
        * AtCoder: https://atcoder.jp/contests/ahc037/tasks/ahc037_a
        *
        * The rectilinear steiner arborescence problem
        * https://link.springer.com/article/10.1007/BF01758762
        *
        */
        std::vector<Point> pos;
        for (int r = 0; r < sparseMatMap.size(); r++)
            for (int c = 0; c < sparseMatMap[r].size(); c++)
                if (sparseMatMap[r][c]) pos.emplace_back(c, r);
        std::vector<StStep> plan;
        while (pos.size() > 1)
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
                plan.emplace_back(Fill, pos[maxI].y, col, Dim::Col, pow2Rects[pos[maxI].y][col]);
            for (int row = pos[maxI].y - 1; row >= maxY; row--)
                plan.emplace_back(Fill, row, maxX, Dim::Row, pow2Rects[row][maxX]);
            for (int col = pos[maxJ].x - 1; col >= maxX; col--)
                plan.emplace_back(Fill, pos[maxJ].y, col, Dim::Col, pow2Rects[pos[maxJ].y][col]);
            for (int row = pos[maxJ].y - 1; row >= maxY; row--)
                plan.emplace_back(Fill, row, maxX, Dim::Row, pow2Rects[row][maxX]);

            pos[maxI] = Point(maxX, maxY);
            swap(pos[maxJ], pos[pos.size() - 1]);
            pos.pop_back();
        }

        reverse(plan.begin(), plan.end());
        return plan;
    }
    case StStrategy::SaveMemory:
    {
        // todo: implement.
        std::vector<StStep> plan;
        return plan;
    }
    }
}

template <typename T>
void morphOp(Op minmax, InputArray _src, OutputArray _dst, InputArray kernel,
    Point anchor, int iterations,
    int borderType, const Scalar& borderVal)
{
    T nil = (minmax == Op::Min) ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();

    Mat _kernel = kernel.getMat();
    int rowDepthLim = log2(longestRowRunLength(_kernel)) + 1;
    int colDepthLim = log2(longestColRunLength(_kernel)) + 1;
    std::vector<std::vector<std::vector<Point>>> pow2Rects
        = genPow2RectsToCoverKernel(_kernel, rowDepthLim, colDepthLim);
    std::vector<StStep> stPlan
        = planSparseTableConstr(pow2Rects, rowDepthLim, colDepthLim, Faster);

    Mat src = _src.getMat();
    _dst.create(_src.size(), _src.type());
    Mat dst = _dst.getMat();

    Scalar bV = borderVal;
    if (borderType == BorderTypes::BORDER_CONSTANT && borderVal == morphologyDefaultBorderValue())
        bV = Scalar::all(nil);

    do
    {
        Mat expandedSrc;
        copyMakeBorder(src, expandedSrc,
            anchor.y, _kernel.cols - 1 - anchor.y,
            anchor.x, _kernel.rows - 1 - anchor.x,
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
        for (int r = 0; r < pow2Rects.size(); r++)
        {
            for (int c = 0; c < pow2Rects[r].size(); c++)
            {
                for (Point p : pow2Rects[r][c])
                {
                    Rect rect(p, Size(1 << c, 1 << r));
                    Rect srcRect(p, dst.size());
                    Mat& srcMat = st[r][c](srcRect);
                    if (minmax == Op::Min) cv::min(dst, srcMat, dst);
                    else cv::max(dst, srcMat, dst);
                }
            }
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
        stMorph::erode(src, dst, kernel, anchor, iterations, borderType, borderVal);
        break;
    case MORPH_DILATE:
        stMorph::dilate(src, dst, kernel, anchor, iterations, borderType, borderVal);
        break;
    case MORPH_OPEN:
        stMorph::erode(src, dst, kernel, anchor, iterations, borderType, borderVal);
        stMorph::dilate(dst, dst, kernel, anchor, iterations, borderType, borderVal);
        break;
    case MORPH_CLOSE:
        stMorph::dilate(src, dst, kernel, anchor, iterations, borderType, borderVal);
        stMorph::erode(dst, dst, kernel, anchor, iterations, borderType, borderVal);
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
