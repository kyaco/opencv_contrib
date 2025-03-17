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
Mat kn5() { return getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE, Size(5, 5)); }
Mat kn4() { return getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE, Size(4, 4)); }
Mat kn1Zero() { return Mat::zeros(1, 1, CV_8UC1); }
Mat kn1One() { return Mat::ones(1, 1, CV_8UC1); }
Mat knEmpty() { return Mat(); }
Mat knZeros() { return Mat::zeros(5, 5, CV_8UC1); }
Mat knOnes() { return Mat::ones(5, 5, CV_8UC1); }
Mat knBig() { return getStructuringElement(cv::MorphShapes::MORPH_RECT, Size(201, 201)); }
Mat knAsymm (){
    return (Mat_<uchar>(5, 5) << 0,0,0,0,0, 0,0,1,0,0, 0,1,0,0,0, 0,0,0,0,0, 0,0,1,0,0);
}
Mat knRnd(int size)
{
    Mat rndMat(size, size, CV_8UC1);
    randu(rndMat, 0, 2);
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

void p2RCov(InputArray kernel)
{
    std::vector<Rect> p2Rects = stMorph::genPow2RectsToCoverKernel(kernel);
    Mat expected = kernel.getMat();
    Mat actual = Mat::zeros(kernel.size(), kernel.type());
    for (Rect p2Rect: p2Rects)
    {
        Rect rect(p2Rect.x, p2Rect.y, 1 << p2Rect.width, 1 << p2Rect.height);
        actual(rect).setTo(1);
    }
    assertArraysIdentical(expected, actual);
}
TEST(ximgproc_StMorph_private, feature_P2RCov_rnd) { p2RCov(knRnd(100)); }

#pragma endregion

#pragma region morph_dev

TEST(ximgproc_StMorph_dev, compare_with_original_erode)
{
    // preparation
    int kRadius = 15;
    //Size sz(200, 150);
    Size sz = szVGA;
    int type = CV_8UC3;

    int kSize = kRadius * 2 + 1;
    Point anchor(kRadius, kRadius);
    Mat src(sz, type);
    Mat expected(sz, type);
    Mat actual(sz, type);
    Size kernelSize(kSize, kSize);
    Mat kernel = getStructuringElement(cv::MorphShapes::MORPH_RECT, kernelSize, Point(kRadius, kRadius));

    src.setTo(240);
    putText(src, "A", Point(sz.height / 5 * 1, sz.height / 20 * 15), HersheyFonts::FONT_HERSHEY_TRIPLEX, 10, Scalar(255, 40, 40), 30, LineTypes::FILLED);
    putText(src, "B", Point(sz.height / 5 * 2, sz.height / 20 * 16), HersheyFonts::FONT_HERSHEY_TRIPLEX, 10, Scalar(20, 255, 0), 30, LineTypes::FILLED);
    putText(src, "C", Point(sz.height / 5 * 3, sz.height / 20 * 17), HersheyFonts::FONT_HERSHEY_TRIPLEX, 10, Scalar(10, 10, 255), 30, LineTypes::FILLED);

    cv::TickMeter timer;

    // original
    timer.start();
    cv::erode(src, expected, kernel); // 482ms for Elipse, kSize = 101
    timer.stop();
    double originalTime = timer.getTimeMilli();
    timer.reset();

    // proposal
    timer.start();
    stMorph::erode(src, actual, kernel); // 217ms for Elipse, kSize = 101
    timer.stop();
    double proposalTime = timer.getTimeMilli();

    // assertion
    Mat diff;
    cv::absdiff(expected, actual, diff);

#if 0
    putText(expected, std::to_string(originalTime), cv::Point(10, 20), HersheyFonts::FONT_HERSHEY_TRIPLEX, 1, Scalar(250, 40, 40), 1, LineTypes::FILLED);
    putText(actual, std::to_string(proposalTime), cv::Point(10, 20), HersheyFonts::FONT_HERSHEY_TRIPLEX, 1, Scalar(250, 40, 40), 1, LineTypes::FILLED);
    Mat con;
    double rate = 300.0 / src.cols;
    cv::hconcat(src, expected, con);
    cv::hconcat(con, actual, con);
    cv::resize(con, con, Size(), rate, rate, InterpolationFlags::INTER_NEAREST);
    cv::resize(diff, diff, Size(), rate, rate, InterpolationFlags::INTER_NEAREST);
    imshow("Bordered source", con);
    imshow("diff", diff);
    waitKey();
    destroyAllWindows();
#endif

    double min, max;
    cv::minMaxLoc(diff, &min, &max);
    CV_Assert(max == 0);
}

TEST(ximgproc_StMorph_dev, POW2RECT_COVERING)
{
    uchar ary[]{
        0, 1, 0, 1, 0, 1, 0, 1,
        1, 1, 0, 1, 1, 1, 1, 1,
        1, 0, 0, 1, 0, 1, 0, 1,
        0, 0, 1, 1, 1, 1, 1, 1,
        0, 1, 0, 1, 0, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        0, 1, 0, 1, 0, 1, 0, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
    };
    Mat kernel(8, 8, CV_8UC1, ary);
    std::vector<Rect> rects = stMorph::genPow2RectsToCoverKernel(kernel);

    int rate = 20;
    resize(kernel * 255, kernel, Size(), rate, rate, InterpolationFlags::INTER_NEAREST);
    cvtColor(kernel, kernel, cv::COLOR_GRAY2BGR);
    Scalar color[20]{
        Scalar(83, 89, 73), Scalar(49, 238, 73), Scalar(220, 192, 189), Scalar(174, 207, 34),
        Scalar(144, 169, 187), Scalar(137, 94, 76), Scalar(42, 11, 215), Scalar(113, 11, 204),
        Scalar(71, 124, 8), Scalar(192, 38, 8), Scalar(82, 201, 8), Scalar(70, 7, 112),
        Scalar(166, 219, 201), Scalar(154, 173, 0), Scalar(132, 127, 139), Scalar(154, 1, 68),
        Scalar(231, 131, 56), Scalar(206, 238, 136), Scalar(188, 78, 173), Scalar(27, 178, 206)
    };
    for (int i = 0; i < rects.size(); i++)
    {
        Rect rect = rects[i];
        Point lt((rect.x) * rate + i % 11, (rect.y) * rate + i % 11);
        Point lb((rect.x) * rate + i % 11, (rect.y + (1 << rect.height)) * rate - 11 + i % 11);
        Point rb((rect.x + (1 << rect.width)) * rate - 11 + i % 11, (rect.y + (1 << rect.height)) * rate - 11 + i % 11);
        Point rt((rect.x + (1 << rect.width)) * rate - 11 + i % 11, (rect.y) * rate + i % 11);
        cv::line(kernel, lt, lb, color[i % 20], 2);
        cv::line(kernel, lb, rb, color[i % 20], 2);
        cv::line(kernel, rb, rt, color[i % 20], 2);
        cv::line(kernel, rt, lt, color[i % 20], 2);
    }
    //imshow("kernel", kernel);

    waitKey();
    destroyAllWindows();
}

TEST(ximgproc_StMorph_dev, PLANNING)
{
    std::vector<std::vector<bool>> map{
        std::vector<bool>{0,0,0,0,0,0,0,1},
        std::vector<bool>{0,0,0,0,0,0,1,0},
        std::vector<bool>{0,0,0,0,0,1,0,0},
        std::vector<bool>{0,0,0,0,1,0,0,0},
        std::vector<bool>{0,0,0,1,0,0,0,0},
        std::vector<bool>{0,0,1,0,0,0,0,0},
        std::vector<bool>{0,1,0,0,0,0,0,0},
        std::vector<bool>{1,0,0,0,0,0,0,0},
    };
    auto res = stMorph::planSparseTableConstr(map);

    int g = 30;
    int r = map.size();
    int c = map[0].size();
    Mat m = Mat::zeros(r * g, c * g, CV_8UC3);
    for (int row = 0; row < r; row++)
    {
        for (int col = 0; col < c; col++)
        {
            if (map[row][col]) cv::rectangle(m, Rect(col * g + g / 2 - 5, row * g + g / 2 - 5, 11, 11), Scalar(20, 20, 255), -1);
        }
    }
    for (int i = 0; i < res.size(); i++)
    {
        auto edge = res[i];
        if (edge.ax == stMorph::Dim::Row)
        {
            cv::line(m, Point(edge.dimCol * g + g / 2, edge.dimRow * g + g / 2), Point(edge.dimCol * g + g / 2, (edge.dimRow + 1) * g + g / 2), Scalar(100, 100, 100), 2);
        }
        else
        {
            cv::line(m, Point(edge.dimCol * g + g / 2, edge.dimRow * g + g / 2), Point((edge.dimCol + 1) * g + g / 2, edge.dimRow * g + g / 2), Scalar(100, 100, 100), 2);
        }
    }
//    imshow("Map", m);

    waitKey();
    destroyAllWindows();
}

void cvDilate(InputArray src, InputArray kernel, Point anchor = Point(-1, -1),
    int iterations = 1,
    BorderTypes bdrType = BorderTypes::BORDER_CONSTANT, Scalar& bdrVal = Scalar::all(DBL_MAX))
{
    Mat actual;
    dilate(src, actual, kernel, anchor, iterations, bdrType, bdrVal);
}
void cvErode(InputArray src, InputArray kernel, Point anchor = Point(-1, -1),
    int iterations = 1,
    BorderTypes bdrType = BorderTypes::BORDER_CONSTANT, Scalar& bdrVal = Scalar::all(DBL_MAX))
{
    Mat actual;
    erode(src, actual, kernel, anchor, iterations, bdrType, bdrVal);
}
Mat kn51() { return getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE, Size(51, 51)); }
TEST(ximgproc_StMorph_dev, big_stDilate)
{
    dilate_ftr(im(CV_8UC3), kn51());
}
TEST(ximgproc_StMorph_dev, big_stEerode)
{
    erode_ftr(im(CV_8UC3), kn51());
}
TEST(ximgproc_StMorph_dev, big_cvDilate)
{
    cvDilate(im(CV_8UC3), kn51());
}
TEST(ximgproc_StMorph_dev, big_cvErode)
{
    cvErode(im(CV_8UC3), kn51());
}

#pragma endregion

}} // ::opencv_test::
