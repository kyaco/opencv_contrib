// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"
#include "opencv2/ximgproc/sparse_table_morphology.hpp"
#include "opencv2/imgproc.hpp"

namespace opencv_test {
namespace st_morphology {

TEST(ximgproc_SparseTableMorph, compare_with_original_erode)
{
    // preparation
    int kRadius = 100;
    Size sz(200, 150);
    // Size sz = szVGA;
    int type = CV_8UC3;

    int kSize = kRadius * 2 + 1;
    Point anchor(kRadius, kRadius);
    Mat src(sz, type);
    Mat expected(sz, type);
    Mat actual(sz, type);
    Size kernelSize(kSize, kSize);
    Mat kernel = getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE, kernelSize, Point(kRadius, kRadius));

    src.setTo(240);
    putText(src, "A", Point(sz.height / 5 * 1, sz.height / 20 * 15), HersheyFonts::FONT_HERSHEY_TRIPLEX, 10, Scalar(250, 40, 40), 30, LineTypes::FILLED);
    putText(src, "B", Point(sz.height / 5 * 2, sz.height / 20 * 16), HersheyFonts::FONT_HERSHEY_TRIPLEX, 10, Scalar(20, 230, 0), 30, LineTypes::FILLED);
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
    ximgproc::st::erode(src, actual, kernel); // 217ms for Elipse, kSize = 101
    timer.stop();
    double proposalTime = timer.getTimeMilli();

    // assertion
    Mat diff;
    cv::absdiff(expected, actual, diff);

#if 1
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

} //
} // opencv_test
