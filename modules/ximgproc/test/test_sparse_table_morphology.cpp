// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "opencv2/ximgproc/sparse_table_morphology.hpp"
#include "opencv2/imgproc.hpp"

namespace opencv_test {
namespace stMorph {

TEST(ximgproc_SparseTableMorph, compare_with_original_erode)
{
    // preparation
    int kRadius = 10;
    //Size sz(200, 150);
     Size sz = szVGA;
    int type = CV_8UC3;

    int kSize = kRadius * 2 + 1;
    Point anchor(kRadius, kRadius);
    Mat src(sz, type);
    Mat expected(sz, type);
    Mat actual(sz, type);
    Size kernelSize(kSize, kSize);
    Mat kernel = getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE, kernelSize, Point(kRadius, kRadius));

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
    ximgproc::stMorph::erode(src, actual, kernel); // 217ms for Elipse, kSize = 101
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

TEST(develop, POW2RECT_COVERING)
{
    int kSize = 11;
    Size kernelSize(kSize, kSize);
    //Mat kernel = getStructuringElement(MorphShapes::MORPH_ELLIPSE, kernelSize);
    uchar ary[] {
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
    std::vector<Rect> rects = ximgproc::stMorph::genPow2RectsToCoverKernel(kernel);

    int rate = 40;
    resize(kernel * 255, kernel, Size(), rate, rate, InterpolationFlags::INTER_NEAREST);
    cvtColor(kernel, kernel, cv::COLOR_GRAY2BGR);
    Scalar color[20]{
        Scalar(83, 89, 73), Scalar(49, 238, 73), Scalar(220, 192, 189), Scalar(174, 207, 34), Scalar(144, 169, 187),
        Scalar(137, 94, 76), Scalar(42, 11, 215), Scalar(113, 11, 204), Scalar(71, 124, 8), Scalar(192, 38, 8),
        Scalar(82, 201, 8), Scalar(70, 7, 112), Scalar(166, 219, 201), Scalar(154, 173, 0), Scalar(132, 127, 139),
        Scalar(154, 1, 68), Scalar(231, 131, 56), Scalar(206, 238, 136), Scalar(188, 78, 173), Scalar(27, 178, 206)
    };
    for (int i = 0; i < rects.size(); i++)
    {
        Rect rect = rects[i];
        Point lt((rect.x                    ) * rate      + i % 11, (rect.y                     ) * rate      + i % 11);
        Point lb((rect.x                    ) * rate      + i % 11, (rect.y + (1 << rect.height)) * rate - 11 + i % 11);
        Point rb((rect.x + (1 << rect.width)) * rate - 11 + i % 11, (rect.y + (1 << rect.height)) * rate - 11 + i % 11);
        Point rt((rect.x + (1 << rect.width)) * rate - 11 + i % 11, (rect.y                     ) * rate      + i % 11);
        cv::line(kernel, lt, lb, color[i % 20], 2);
        cv::line(kernel, lb, rb, color[i % 20], 2);
        cv::line(kernel, rb, rt, color[i % 20], 2);
        cv::line(kernel, rt, lt, color[i % 20], 2);
    }
    imshow("kernel", kernel);

    waitKey();
    destroyAllWindows();
}

TEST(develop, PLANNING)
{
    std::vector<std::vector<bool>> map{
        std::vector<bool>{0,0,0,0,0,0,0,1},
        std::vector<bool>{0,0,0,0,0,0,1,0},
        std::vector<bool>{0,0,0,0,0,1,0,0},
        std::vector<bool>{0,0,0,0,1,0,0,0},
        std::vector<bool>{0,0,0,0,0,1,0,0},
        std::vector<bool>{0,0,1,0,0,0,0,0},
        std::vector<bool>{0,1,0,0,1,0,0,0},
        std::vector<bool>{1,0,0,0,0,0,0,0},
    };
    auto res = ximgproc::stMorph::planSparseTableConstruction(map);

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
        if (edge.ax == ximgproc::stMorph::Dim::Row)
        {
            cv::line(m, Point(edge.dimCol * g + g / 2, edge.dimRow * g + g / 2), Point(edge.dimCol * g + g / 2, (edge.dimRow + 1) * g + g / 2), Scalar(100, 100, 100), 2);
        }
        else
        {
            cv::line(m, Point(edge.dimCol * g + g / 2, edge.dimRow * g + g / 2), Point((edge.dimCol + 1) * g + g / 2, edge.dimRow * g + g / 2), Scalar(100, 100, 100), 2);
        }
    }
    imshow("Map", m);

    waitKey();
    destroyAllWindows();
}

} //
} // opencv_test
