// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include <limits>
#include <utility>
#include <vector>

namespace cv {
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

enum Op
{
    Min, Max
};

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

std::vector<Point> findP2RectCorners(const Mat& stNode, int rowDepth, int colDepth)
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

            // ignore if neighboring block is white
            if (col + colOfst < stNode.cols && stNode.at<uchar>(row, col + colOfst) == 1) continue;
            if (col - colOfst >= 0 && stNode.at<uchar>(row, col - colOfst) == 1) continue;
            if (row + rowOfst < stNode.rows && stNode.at<uchar>(row + rowOfst, col) == 1) continue;
            if (row - rowOfst >= 0 && stNode.at<uchar>(row - rowOfst, col) == 1) continue;

            p2Rects.emplace_back(col, row);
        }
    }
    return p2Rects;
}

/*
* Find a set of power-2-rectangles to cover the kernel.
* power-2-rectangles is a rectangle whose height and width are both power of 2.
*/
std::vector<std::vector<std::vector<Point>>> genPow2RectsToCoverKernel(
    const Mat& kernel, int rowDepthLim, int colDepthLim)
{
    CV_Assert(kernel.type() == CV_8UC1);

    std::vector<std::vector<std::vector<Point>>> p2Rects;
    Mat stNodeCache = kernel;
    for (int rowDepth = 0; rowDepth < rowDepthLim; rowDepth++)
    {
        Mat stNode = stNodeCache.clone();
        p2Rects.emplace_back(std::vector<std::vector<Point>>());
        for (int colDepth = 0; colDepth < colDepthLim; colDepth++)
        {
            p2Rects[rowDepth].emplace_back(findP2RectCorners(stNode, rowDepth, colDepth));
            int colStep = 1 << colDepth;
            if (stNode.cols - colStep < 0) break;
            Rect s1(0, 0, stNode.cols - colStep, stNode.rows);
            Rect s2(colStep, 0, stNode.cols - colStep, stNode.rows);
            cv::min(stNode(s1), stNode(s2), stNode);
        }
        int rowStep = 1 << rowDepth;
        if (stNodeCache.rows - rowStep < 0) break;
        Rect s1(0, 0, stNodeCache.cols, stNodeCache.rows - rowStep);
        Rect s2(0, rowStep, stNodeCache.cols, stNodeCache.rows - rowStep);
        cv::min(stNodeCache(s1), stNodeCache(s2), stNodeCache);
    }

    return p2Rects;
}

Mat SolveRSAPGreedy(const Mat& initialMap)
{
    /*
    * Solves the rectilinear steiner arborescence problem greedy.
    * https://link.springer.com/article/10.1007/BF01758762
    *
    * Following implementation is O(n^3)-time algorithm
    * which is different from the mothod proposed in the paper.
    */
    CV_Assert(initialMap.type() == CV_8UC1);
    std::vector<Point> pos;
    for (int r = 0; r < initialMap.rows; r++)
        for (int c = 0; c < initialMap.cols; c++)
            if (initialMap.at<uchar>(r, c) == 1) pos.emplace_back(c, r);
    Mat resMap = Mat::zeros(initialMap.size(), CV_8UC2);

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
            resMap.at<Vec2b>(pos[maxI].y, col)[1] = 1;
        for (int row = pos[maxI].y - 1; row >= maxY; row--)
            resMap.at<Vec2b>(row, maxX)[0] = 1;
        for (int col = pos[maxJ].x - 1; col >= maxX; col--)
            resMap.at<Vec2b>(pos[maxJ].y, col)[1] = 1;
        for (int row = pos[maxJ].y - 1; row >= maxY; row--)
            resMap.at<Vec2b>(row, maxX)[0] = 1;

        pos[maxI] = Point(maxX, maxY);
        swap(pos[maxJ], pos[pos.size() - 1]);
        pos.pop_back();
    }
    return resMap;
}

Mat sparseTableFillPlanning(
    std::vector<std::vector<std::vector<Point>>> pow2Rects, int rowDepthLim, int colDepthLim)
{
    /*
    * Plan the order to fill the required 2d-sparse-table nodes.
    * The type of returned mat is Vec2b.
    * if path[dr][dc][0] == 1 then st[dr+1][dc] will be calculated from st[dr][dc].
    * if path[dr][dc][1] == 1 then st[dr][dc+1] will be calculated from st[dr][dc].
    */

    // list up required sparse table nodes.
    Mat stMap = Mat::zeros(rowDepthLim, colDepthLim, CV_8UC1);
    for (int rd = 0; rd < rowDepthLim; rd++)
        for (int cd = 0; cd < colDepthLim; cd++)
            if (pow2Rects[rd][cd].size() > 0)
                stMap.at<uchar>(rd, cd) = 1;
    stMap.at<uchar>(0, 0) = 1;
    Mat path = SolveRSAPGreedy(stMap);
    return path;
}

void morphDfs(int minmax, Mat& st, Mat& dst,
    std::vector<std::vector<std::vector<Point>>> row2Rects, const Mat& stPlan,
    int rowDepth, int colDepth)
{
    for (Point p : row2Rects[rowDepth][colDepth])
    {
        Rect rect(p, dst.size());
        if (minmax == Op::Min) cv::min(dst, st(rect), dst);
        else cv::max(dst, st(rect), dst);
    }

    if (stPlan.at<Vec2b>(rowDepth, colDepth)[1] == 1)
    {
        // col direction
        Mat st2 = st;
        int ofs = 1 << colDepth;
        Rect rect1(0, 0, st2.cols - ofs, st2.rows);
        Rect rect2(ofs, 0, st2.cols - ofs, st2.rows);

        if (minmax == Op::Min) cv::min(st2(rect1), st2(rect2), st2);
        else cv::max(st2(rect1), st2(rect2), st2);
        morphDfs(minmax, st2, dst, row2Rects, stPlan, rowDepth, colDepth + 1);
    }
    if (stPlan.at<Vec2b>(rowDepth, colDepth)[0] == 1)
    {
        // row direction
        int ofs = 1 << rowDepth;
        Rect rect1(0, 0, st.cols, st.rows - ofs);
        Rect rect2(0, ofs, st.cols, st.rows - ofs);

        if (minmax == Op::Min) cv::min(st(rect1), st(rect2), st);
        else cv::max(st(rect1), st(rect2), st);
        morphDfs(minmax, st, dst, row2Rects, stPlan, rowDepth + 1, colDepth);
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
    Mat stPlan
        = sparseTableFillPlanning(pow2Rects, rowDepthLim, colDepthLim);

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
        morphDfs(minmax, expandedSrc, dst, pow2Rects, stPlan, 0, 0);
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

}} // cv::stMorph::
