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

// this method may be applied for the covering polygon problem with rectangle.
// https://www.sciencedirect.com/science/article/pii/S0019995884800121
std::tuple<std::vector<std::vector<Mat>>, std::vector<Rect>> genPow2RectsToCoverKernel_dev(InputArray _kernel)
{
    CV_Assert(_kernel.type() == CV_8UC1);

    Mat kernel = _kernel.getMat();

    // generate log2 table
    int len = max(kernel.rows, kernel.cols) + 1;
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

        st[0][colDepth] = Mat::zeros(kernel.rows, kernel.cols, kernel.type());
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

            st[rowDepth][colDepth] = Mat::zeros(kernel.rows, kernel.cols, kernel.type());
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
        int rowSkip = (1 << rowDepth) - 1;
        int rowLim = kernel.rows - rowSkip;
        for (int colDepth = 0; colDepth <= log2[kernel.cols]; colDepth++)
        {
            int colSkip = (1 << colDepth) - 1;
            int colLim = kernel.cols - colSkip;

            uchar* ptr = st[rowDepth][colDepth].ptr();
            for (int row = 0; row < rowLim; row++)
            {
                for (int col = 0; col < colLim; col++, ptr++)
                {
                    // ignore black cell
                    if (ptr[0] == 0) continue;

                    // ignore if both sides are white by each axis
                    if (col > 0 && ptr[-1] == 1 && col < colLim && ptr[1] == 1) continue;
                    if (row > 0 && ptr[-kernel.cols] && row < rowLim && ptr[kernel.cols] == 1) continue;

                    // ignore one of neighbor block is white; will be alive in deeper table
                    if (col + (1 << colDepth) <= colLim && ptr[1 << colDepth] == 1) continue;
                    if (col - (1 << colDepth) >= 0 && ptr[-(1 << colDepth)] == 1) continue;
                    if (row + (1 << rowDepth) <= rowLim && ptr[(1 << rowDepth) * kernel.cols] == 1) continue;
                    if (row - (1 << rowDepth) >= 0 && ptr[-(1 << rowDepth) * kernel.cols] == 1) continue;

                    p2Rects.emplace_back(col, row, colDepth, rowDepth);
                }
                ptr += colSkip;
            }
        }
    }

    return { st, p2Rects };
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

    std::tuple<std::vector<std::vector<Mat>>, std::vector<Rect>> ret = genPow2RectsToCoverKernel_dev(kernel);

    // visualize sparse table
    std::vector<std::vector<Mat>> st = std::get<0>(ret);
    int cellSize = 16;
    Mat concatSt;
    std::vector<Mat> hconMat(st.size(), Mat());
    for (int row = 0; row < st.size(); row++) for (int col = 0; col < st[row].size(); col++)
    {
        Mat t = st[row][col];
        Mat x = Mat::zeros(t.rows * cellSize, t.cols * cellSize, CV_8UC3);
        uchar* pCell = t.ptr();
        for (int r = 0; r < t.rows; r++) for (int c = 0; c < t.cols; c++, pCell++)
        {
            if (*pCell == 1) cv::rectangle(x, Rect(c * cellSize, r * cellSize, cellSize, cellSize), Scalar(255, 255, 255), -1);
        }
        for (int r = 1; r < t.rows; r++) cv::line(x, Point(0, r * cellSize), Point(x.cols, r * cellSize), Scalar(50, 50, 50), 1);
        for (int c = 1; c < t.cols; c++) cv::line(x, Point(c * cellSize, 0), Point(c * cellSize, x.rows), Scalar(50, 50, 50), 1);
        cv::copyMakeBorder(x, x, 0, 2, 0, 2, BorderTypes::BORDER_CONSTANT, Scalar(200, 200, 200));
        st[row][col] = x;
    }
    for (int row = 0; row < st.size(); row++) hconcat(st[row], hconMat[row]); vconcat(hconMat, concatSt);
    imshow("result", concatSt);

    // visualize rectangles on kernel
    std::vector<Rect> rects = std::get<1>(ret);
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

} //
} // opencv_test
