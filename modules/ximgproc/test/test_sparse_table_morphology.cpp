// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "opencv2/ximgproc/sparse_table_morphology.hpp"
#include <vector>

namespace opencv_test {
namespace {

#pragma region Common test methods

void assertArraysIdentical(InputArray ary1, InputArray ary2)
{
    Mat xor = ary1.getMat() ^ ary2.getMat();
    ASSERT_EQ(cv::countNonZero(xor.reshape(1)), 0);
}
Mat im(int type)
{
    int depth = CV_MAT_DEPTH(type);
    int ch = CV_MAT_CN(type);
    Mat img = imread(cvtest::TS::ptr()->get_data_path() + "cv/shared/lena.png");
    // ASSERT_EQ(img.type(), CV_8UC3);

    if (ch == 1) cv::cvtColor(img, img, ColorConversionCodes::COLOR_BGR2GRAY, ch);
    if (depth == CV_8S) img /= 2;
    img.convertTo(img, depth);
    if (depth == CV_16S) img *= (1 << 7);
    if (depth == CV_16U) img *= (1 << 8);
    if (depth == CV_32S) img *= (1 << 23);
    if (depth == CV_32F) img /= (1 << 8);
    if (depth == CV_64F) img /= (1 << 8);

    return img;
}
Mat kn4() { return getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE, Size(4, 4)); }
Mat kn5() { return getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE, Size(5, 5)); }
Mat kn51() { return getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE, Size(51, 51)); }
Mat knBig() { return getStructuringElement(cv::MorphShapes::MORPH_RECT, Size(201, 201)); }
Mat kn1Zero() { return Mat::zeros(1, 1, CV_8UC1); }
Mat kn1One() { return Mat::ones(1, 1, CV_8UC1); }
Mat knEmpty() { return Mat(); }
Mat knZeros() { return Mat::zeros(5, 5, CV_8UC1); }
Mat knOnes() { return Mat::ones(5, 5, CV_8UC1); }
Mat knAsymm (){ return (Mat_<uchar>(5, 5) << 0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0); }
Mat knRnd(int size, int density)
{
    Mat rndMat(size, size, CV_8UC1);
    theRNG().state = getTickCount();
    randu(rndMat, 2, 102);
    density++;
    rndMat.setTo(0, density < rndMat);
    rndMat.setTo(1, 1 < rndMat);
    return rndMat;
}

#pragma endregion

#pragma region dilasion

/*
* dilate regression tests.
*/
void dilate_rgr(InputArray src, InputArray kernel, Point anchor = Point(-1, -1),
    int iterations = 1,
    BorderTypes bdrType = BorderTypes::BORDER_CONSTANT, Scalar& bdrVal = Scalar::all(DBL_MAX))
{
    Mat expected, actual;
    dilate(src, expected, kernel, anchor, iterations, bdrType, bdrVal);
    stMorph::dilate(src, actual, kernel, anchor, iterations, bdrType, bdrVal);
    assertArraysIdentical(expected, actual);
}
TEST(ximgproc_StMorph_dilate, regression_8UC1) { dilate_rgr(im(CV_8UC1), kn5()); }
TEST(ximgproc_StMorph_dilate, regression_8UC3) { dilate_rgr(im(CV_8UC3), kn5()); }
TEST(ximgproc_StMorph_dilate, regression_16UC1) { dilate_rgr(im(CV_16UC1), kn5()); }
TEST(ximgproc_StMorph_dilate, regression_16UC3) { dilate_rgr(im(CV_16UC3), kn5()); }
TEST(ximgproc_StMorph_dilate, regression_16SC1) { dilate_rgr(im(CV_16SC1), kn5()); }
TEST(ximgproc_StMorph_dilate, regression_16SC3) { dilate_rgr(im(CV_16SC3), kn5()); }
TEST(ximgproc_StMorph_dilate, regression_32FC1) { dilate_rgr(im(CV_32FC1), kn5()); }
TEST(ximgproc_StMorph_dilate, regression_32FC3) { dilate_rgr(im(CV_32FC3), kn5()); }
TEST(ximgproc_StMorph_dilate, regression_64FC1) { dilate_rgr(im(CV_64FC1), kn5()); }
TEST(ximgproc_StMorph_dilate, regression_64FC3) { dilate_rgr(im(CV_64FC3), kn5()); }
TEST(ximgproc_StMorph_dilate, regression_kn5) { dilate_rgr(im(CV_8UC3), kn5()); }
TEST(ximgproc_StMorph_dilate, regression_kn4) { dilate_rgr(im(CV_8UC3), kn4()); }
TEST(ximgproc_StMorph_dilate, wtf_regression_kn1Zero) { dilate_rgr(im(CV_8UC3), kn1Zero()); }
TEST(ximgproc_StMorph_dilate, regression_kn1One) { dilate_rgr(im(CV_8UC3), kn1One()); }
TEST(ximgproc_StMorph_dilate, wtf_regression_knEmpty) { dilate_rgr(im(CV_8UC3), knEmpty()); }
TEST(ximgproc_StMorph_dilate, wtf_regression_knZeros) { dilate_rgr(im(CV_8UC3), knZeros()); }
TEST(ximgproc_StMorph_dilate, regression_knOnes) { dilate_rgr(im(CV_8UC3), knOnes()); }
TEST(ximgproc_StMorph_dilate, regression_knBig) { dilate_rgr(im(CV_8UC3), knBig()); }
TEST(ximgproc_StMorph_dilate, regression_knAsymm) { dilate_rgr(im(CV_8UC3), knAsymm()); }
TEST(ximgproc_StMorph_dilate, regression_ancMid) { dilate_rgr(im(CV_8UC3), kn5(), Point(-1, -1)); }
TEST(ximgproc_StMorph_dilate, regression_ancEdge1) { dilate_rgr(im(CV_8UC3), kn5(), Point(0, 0)); }
TEST(ximgproc_StMorph_dilate, regression_ancEdge2) { dilate_rgr(im(CV_8UC3), kn5(), Point(4, 4)); }
TEST(ximgproc_StMorph_dilate, wtf_regression_it0) { dilate_rgr(im(CV_8UC3), kn5(), Point(-1, -1), 0); }
TEST(ximgproc_StMorph_dilate, regression_it1) { dilate_rgr(im(CV_8UC3), kn5(), Point(-1, -1), 1); }
TEST(ximgproc_StMorph_dilate, regression_it2) { dilate_rgr(im(CV_8UC3), kn5(), Point(-1, -1), 2); }
/*
* dilate feature tests.
*/
void dilate_ftr(InputArray src, InputArray kernel, Point anchor = Point(-1, -1),
    int iterations = 1,
    BorderTypes bdrType = BorderTypes::BORDER_CONSTANT, Scalar& bdrVal = Scalar::all(DBL_MAX))
{
    Mat expected, actual;
    stMorph::dilate(src, actual, kernel, anchor, iterations, bdrType, bdrVal);
    // todo: generate expected result.
    // assertArraysIdentical(expected, actual);
}
/* CV_8S, CV_16F are not supported by morph.simd::getMorphologyFilter */
TEST(ximgproc_StMorph_dilate, feature_8SC1) { dilate_ftr(im(CV_8SC1), kn5()); }
TEST(ximgproc_StMorph_dilate, feature_8SC3) { dilate_ftr(im(CV_8SC3), kn5()); }
TEST(ximgproc_StMorph_dilate, feature_32SC1) { dilate_ftr(im(CV_32SC1), kn5()); }
TEST(ximgproc_StMorph_dilate, feature_32SC3) { dilate_ftr(im(CV_32SC3), kn5()); }

#pragma endregion

#pragma region erosion

/*
* erode regression tests.
*/
void erode_rgr(InputArray src, InputArray kernel, Point anchor = Point(-1, -1),
    int iterations = 1,
    BorderTypes bdrType = BorderTypes::BORDER_CONSTANT, Scalar& bdrVal = Scalar::all(DBL_MAX))
{
    Mat expected, actual;
    erode(src, expected, kernel, anchor, iterations, bdrType, bdrVal);
    stMorph::erode(src, actual, kernel, anchor, iterations, bdrType, bdrVal);
    assertArraysIdentical(expected, actual);
}
TEST(ximgproc_StMorph_erode, regression_8UC1) { erode_rgr(im(CV_8UC1), kn5()); }
TEST(ximgproc_StMorph_erode, regression_8UC3) { erode_rgr(im(CV_8UC3), kn5()); }
TEST(ximgproc_StMorph_erode, regression_16UC1) { erode_rgr(im(CV_16UC1), kn5()); }
TEST(ximgproc_StMorph_erode, regression_16UC3) { erode_rgr(im(CV_16UC3), kn5()); }
TEST(ximgproc_StMorph_erode, regression_16SC1) { erode_rgr(im(CV_16SC1), kn5()); }
TEST(ximgproc_StMorph_erode, regression_16SC3) { erode_rgr(im(CV_16SC3), kn5()); }
TEST(ximgproc_StMorph_erode, regression_32FC1) { erode_rgr(im(CV_32FC1), kn5()); }
TEST(ximgproc_StMorph_erode, regression_32FC3) { erode_rgr(im(CV_32FC3), kn5()); }
TEST(ximgproc_StMorph_erode, regression_64FC1) { erode_rgr(im(CV_64FC1), kn5()); }
TEST(ximgproc_StMorph_erode, regression_64FC3) { erode_rgr(im(CV_64FC3), kn5()); }
TEST(ximgproc_StMorph_erode, regression_kn5) { erode_rgr(im(CV_8UC3), kn5()); }
TEST(ximgproc_StMorph_erode, regression_kn4) { erode_rgr(im(CV_8UC3), kn4()); }
TEST(ximgproc_StMorph_erode, wtf_regression_kn1Zero) { erode_rgr(im(CV_8UC3), kn1Zero()); }
TEST(ximgproc_StMorph_erode, regression_kn1One) { erode_rgr(im(CV_8UC3), kn1One()); }
TEST(ximgproc_StMorph_erode, wtf_regression_knEmpty) { erode_rgr(im(CV_8UC3), knEmpty()); }
TEST(ximgproc_StMorph_erode, wtf_regression_knZeros) { erode_rgr(im(CV_8UC3), knZeros()); }
TEST(ximgproc_StMorph_erode, regression_knOnes) { erode_rgr(im(CV_8UC3), knOnes()); }
TEST(ximgproc_StMorph_erode, regression_knBig) { erode_rgr(im(CV_8UC3), knBig()); }
TEST(ximgproc_StMorph_erode, regression_knAsymm) { erode_rgr(im(CV_8UC3), knAsymm()); }
TEST(ximgproc_StMorph_erode, regression_ancMid) { erode_rgr(im(CV_8UC3), kn5(), Point(-1, -1)); }
TEST(ximgproc_StMorph_erode, regression_ancEdge1) { erode_rgr(im(CV_8UC3), kn5(), Point(0, 0)); }
TEST(ximgproc_StMorph_erode, regression_ancEdge2) { erode_rgr(im(CV_8UC3), kn5(), Point(4, 4)); }
TEST(ximgproc_StMorph_erode, wtf_regression_it0) { erode_rgr(im(CV_8UC3), kn5(), Point(-1, -1), 0); }
TEST(ximgproc_StMorph_erode, regression_it1) { erode_rgr(im(CV_8UC3), kn5(), Point(-1, -1), 1); }
TEST(ximgproc_StMorph_erode, regression_it2) { erode_rgr(im(CV_8UC3), kn5(), Point(-1, -1), 2); }
/*
* erode feature tests.
*/
void erode_ftr(InputArray src, InputArray kernel, Point anchor = Point(-1, -1),
    int iterations = 1,
    BorderTypes bdrType = BorderTypes::BORDER_CONSTANT, Scalar& bdrVal = Scalar::all(DBL_MAX))
{
    Mat expected, actual;
    stMorph::erode(src, actual, kernel, anchor, iterations, bdrType, bdrVal);
    // todo: generate expected result.
    // assertArraysIdentical(expected, actual);
}
/* CV_8S, CV_16F are not supported by morph.simd::getMorphologyFilter */
TEST(ximgproc_StMorph_erode, feature_8SC1) { erode_ftr(im(CV_8SC1), kn5()); }
TEST(ximgproc_StMorph_erode, feature_8SC3) { erode_ftr(im(CV_8SC3), kn5()); }
TEST(ximgproc_StMorph_erode, feature_32SC1) { erode_ftr(im(CV_32SC1), kn5()); }
TEST(ximgproc_StMorph_erode, feature_32SC3) { erode_ftr(im(CV_32SC3), kn5()); }

#pragma endregion

#pragma region morphologyEx

/*
* morphologyEx regression tests.
*/
void ex_rgr(InputArray src, MorphTypes op, InputArray kernel, Point anchor = Point(-1, -1),
    int iterations = 1,
    BorderTypes bdrType = BorderTypes::BORDER_CONSTANT, Scalar& bdrVal = Scalar::all(DBL_MAX))
{
    Mat expected, actual;
    morphologyEx(src, expected, op, kernel, anchor, iterations, bdrType, bdrVal);
    stMorph::morphologyEx(src, actual, op, kernel, anchor, iterations, bdrType, bdrVal);
    assertArraysIdentical(expected, actual);
}
TEST(ximgproc_StMorph_ex, regression_erode) { ex_rgr(im(CV_8UC3), MORPH_ERODE, kn5()); }
TEST(ximgproc_StMorph_ex, regression_dilage) { ex_rgr(im(CV_8UC3), MORPH_DILATE, kn5()); }
TEST(ximgproc_StMorph_ex, regression_open) { ex_rgr(im(CV_8UC3), MORPH_OPEN, kn5()); }
TEST(ximgproc_StMorph_ex, regression_close) { ex_rgr(im(CV_8UC3), MORPH_CLOSE, kn5()); }
TEST(ximgproc_StMorph_ex, regression_gradient) { ex_rgr(im(CV_8UC3), MORPH_GRADIENT, kn5()); }
TEST(ximgproc_StMorph_ex, regression_tophat) { ex_rgr(im(CV_8UC3), MORPH_TOPHAT, kn5()); }
TEST(ximgproc_StMorph_ex, regression_blackhat) { ex_rgr(im(CV_8UC3), MORPH_BLACKHAT, kn5()); }
TEST(ximgproc_StMorph_ex, regression_hitmiss) { ex_rgr(im(CV_8UC1), MORPH_HITMISS, kn5()); }

#pragma endregion

#pragma region power2RectCovering

std::vector<std::vector<std::vector<Point>>> p2RCov(InputArray kernel)
{
    Mat _kernel = kernel.getMat();
    int rowDepthLim = stMorph::log2(stMorph::longestRowRunLength(_kernel)) + 1;
    int colDepthLim = stMorph::log2(stMorph::longestColRunLength(_kernel)) + 1;
    std::vector<std::vector<std::vector<Point>>> p2Rects
        = stMorph::genPow2RectsToCoverKernel(_kernel, rowDepthLim, colDepthLim);
    Mat expected = kernel.getMat();
    Mat actual = Mat::zeros(kernel.size(), kernel.type());
    for (int r = 0; r < p2Rects.size(); r++)
    {
        for (int c = 0; c < p2Rects[r].size(); c++)
        {
            for (Point p : p2Rects[r][c])
            {
                Rect rect(p.x, p.y, 1 << c, 1 << r);
                actual(rect).setTo(1);
            }
        }
    }
    assertArraysIdentical(expected, actual);
    return p2Rects;
}
void VisualizeCovering(Mat& kernel, const std::vector<std::vector<std::vector<Point>>>& rects)
{
    const int rate = 20;
    const int fluct = 5;
    const int colors = 20;
    resize(kernel * 255, kernel, Size(), rate, rate, InterpolationFlags::INTER_NEAREST);
    cvtColor(kernel, kernel, cv::COLOR_GRAY2BGR);
    Scalar color[colors]{
        Scalar(83, 89, 73), Scalar(49, 238, 73), Scalar(220, 192, 189), Scalar(174, 207, 34),
        Scalar(144, 169, 187), Scalar(137, 94, 76), Scalar(42, 11, 215), Scalar(113, 11, 204),
        Scalar(71, 124, 8), Scalar(192, 38, 8), Scalar(82, 201, 8), Scalar(70, 7, 112),
        Scalar(166, 219, 201), Scalar(154, 173, 0), Scalar(132, 127, 139), Scalar(154, 1, 68),
        Scalar(231, 131, 56), Scalar(206, 238, 136), Scalar(188, 78, 173), Scalar(27, 178, 206)
    };
    int i = 0;
    for (int r = 0; r < rects.size(); r++)
    {
        for (int c = 0; c < rects[r].size(); c++)
        {
            Size s(1 << c, 1 << r);
            for (Point p : rects[r][c])
            {
                Rect rect(p, s);
                int l = (rect.x) * rate + i % fluct;
                int t = (rect.y) * rate + i % fluct;
                int r = (rect.x + rect.width) * rate - fluct + i % fluct;
                int b = (rect.y + rect.height) * rate - fluct + i % fluct;
                Point lt(l, t);
                Point lb(l, b);
                Point rb(r, b);
                Point rt(r, t);
                cv::line(kernel, lt, lb, color[i % colors], 1);
                cv::line(kernel, lb, rb, color[i % colors], 1);
                cv::line(kernel, rb, rt, color[i % colors], 1);
                cv::line(kernel, rt, lt, color[i % colors], 1);
                i++;
            }
        }
    }
#if 0
    imshow("Map", kernel);
    waitKey();
    destroyAllWindows();
#endif
}
TEST(ximgproc_StMorph_private, feature_P2RCov_rnd1) { p2RCov(knRnd(1000, 1)); }
TEST(ximgproc_StMorph_private, feature_P2RCov_rnd10) { p2RCov(knRnd(1000, 10)); }
TEST(ximgproc_StMorph_private, feature_P2RCov_rnd30) { p2RCov(knRnd(1000, 30)); }
TEST(ximgproc_StMorph_private, feature_P2RCov_rnd50) { p2RCov(knRnd(1000, 50)); }
TEST(ximgproc_StMorph_private, feature_P2RCov_rnd80) { p2RCov(knRnd(1000, 80)); }
TEST(ximgproc_StMorph_private, feature_P2RCov_rnd90) { p2RCov(knRnd(1000, 90)); }
TEST(ximgproc_StMorph_private, feature_P2RCov_visualize) {
    Mat kernel = knRnd(50, 70);
    auto rects = p2RCov(kernel);
    VisualizeCovering(kernel, rects);
}

#pragma endregion

#pragma region planning

void VisualizePlanning(
    std::vector<std::vector<std::vector<Point>>> map, std::vector<stMorph::StStep> res)
{
    int g = 30;
    int r = map.size();
    int c = map[0].size();
    Mat m = Mat::zeros(r * g, c * g, CV_8UC3);
    for (int row = 0; row < r; row++)
    {
        for (int col = 0; col < c; col++)
        {
            Rect nodeRect(col * g + g / 2 - 5, row * g + g / 2 - 5, 11, 11);
            if (map[row][col].size() > 0)
                cv::rectangle(m, nodeRect, Scalar(20, 20, 255), -1);
        }
    }
    for (int i = 0; i < res.size(); i++)
    {
        auto edge = res[i];
        Point sp(edge.dimCol * g + g / 2, edge.dimRow * g + g / 2);
        Point ep;
        if (edge.ax == stMorph::Dim::Row)
            ep = Point(edge.dimCol * g + g / 2, (edge.dimRow + 1) * g + g / 2);
        else
            ep = Point((edge.dimCol + 1) * g + g / 2, edge.dimRow * g + g / 2);
        cv::line(m, sp, ep, Scalar(100, 100, 100), 2);
    }
#if 0
    imshow("Map", m);
    waitKey();
    destroyAllWindows();
#endif
}
void feture_planning(const Mat& mat)
{
    std::vector<std::vector<std::vector<Point>>> map;
    for (int r = 0; r < mat.rows; r++)
    {
        map.push_back(std::vector<std::vector<Point>>());
        for (int c = 0; c < mat.cols; c++)
        {
            map[r].push_back(std::vector<Point>());
            if (mat.at<uchar>(r, c) == 1)
            {
                map[r][c].push_back(Point(0, 0));
            }
        }
    }

    auto r = stMorph::planSparseTableConstr(map, mat.rows, mat.cols, stMorph::StStrategy::Faster);
    VisualizePlanning(map, r);
}
TEST(ximgproc_StMorph_private, planning1) { feture_planning(knAsymm()); }
TEST(ximgproc_StMorph_private, planning2){ feture_planning(knRnd(14, 20)); }

#pragma endregion

#pragma region morph_comp

void stDilate(InputArray src, InputArray kernel, Point anchor = Point(-1, -1),
    int iterations = 1,
    BorderTypes bdrType = BorderTypes::BORDER_CONSTANT, Scalar& bdrVal = Scalar::all(DBL_MAX))
{
    Mat tmp;
    stMorph::dilate(src, tmp, kernel, anchor, iterations, bdrType, bdrVal);
}
void stErode(InputArray src, InputArray kernel, Point anchor = Point(-1, -1),
    int iterations = 1,
    BorderTypes bdrType = BorderTypes::BORDER_CONSTANT, Scalar& bdrVal = Scalar::all(DBL_MAX))
{
    Mat tmp;
    stMorph::erode(src, tmp, kernel, anchor, iterations, bdrType, bdrVal);
}
void cvDilate(InputArray src, InputArray kernel, Point anchor = Point(-1, -1),
    int iterations = 1,
    BorderTypes bdrType = BorderTypes::BORDER_CONSTANT, Scalar& bdrVal = Scalar::all(DBL_MAX))
{
    Mat tmp;
    dilate(src, tmp, kernel, anchor, iterations, bdrType, bdrVal);
}
void cvErode(InputArray src, InputArray kernel, Point anchor = Point(-1, -1),
    int iterations = 1,
    BorderTypes bdrType = BorderTypes::BORDER_CONSTANT, Scalar& bdrVal = Scalar::all(DBL_MAX))
{
    Mat tmp;
    erode(src, tmp, kernel, anchor, iterations, bdrType, bdrVal);
}
TEST(ximgproc_StMorph_comp, 51_stDilate) { stDilate(im(CV_8UC3), kn51()); }
TEST(ximgproc_StMorph_comp, 51_stEerode) { stErode(im(CV_8UC3), kn51()); }
TEST(ximgproc_StMorph_comp, 51_cvDilate) { cvDilate(im(CV_8UC3), kn51()); }
TEST(ximgproc_StMorph_comp, 51_cvErode) { cvErode(im(CV_8UC3), kn51()); }
TEST(ximgproc_StMorph_comp, 5_stDilate) { stDilate(im(CV_8UC3), knOnes()); }
TEST(ximgproc_StMorph_comp, 5_stEerode) { stErode(im(CV_8UC3), knOnes()); }
TEST(ximgproc_StMorph_comp, 5_cvDilate) { cvDilate(im(CV_8UC3), knOnes()); }
TEST(ximgproc_StMorph_comp, 5_cvErode) { cvErode(im(CV_8UC3), knOnes()); }

#pragma endregion

}} // ::opencv_test::
