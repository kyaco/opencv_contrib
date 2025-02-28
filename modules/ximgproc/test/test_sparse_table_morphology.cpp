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
                    if (ptr[0] == 0) continue;

                    if (0 < row && ptr[-1] == 1
                        && row < rowLim - 1 && ptr[1] == 1
                        && 0 < col && ptr[-1] == 1
                        && col < colLim - 1 && ptr[1] == 1) continue;

                    if (rowDepth < log2[kernel.rows] &&
                        (st[rowDepth + 1][colDepth].ptr(row, col)[0] == 1 ||
                            (row > (1 << rowDepth) - 1 &&
                                st[rowDepth + 1][colDepth].ptr(row - (1 << (rowDepth)), col)[0] == 1
                                )
                            )
                        ) continue;

                    if (colDepth < log2[kernel.cols] &&
                        (st[rowDepth][colDepth + 1].ptr(row, col)[0] == 1 ||
                            (col > (1 << colDepth) - 1 &&
                                st[rowDepth][colDepth + 1].ptr(row, col - (1 << (colDepth)))[0] == 1
                                )
                            )
                        ) continue;

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
    Mat kernel = getStructuringElement(MorphShapes::MORPH_ELLIPSE, kernelSize);

    std::tuple<std::vector<std::vector<Mat>>, std::vector<Rect>> ret = genPow2RectsToCoverKernel_dev(kernel);
    std::vector<std::vector<Mat>> st = std::get<0>(ret);
    std::vector<Rect> rects = std::get<1>(ret);

    // visualize
    Mat concatSt;
    std::vector<Mat> hconMat(st.size(), Mat());
    for (int row = 0; row < st.size(); row++) hconcat(st[row], hconMat[row]); vconcat(hconMat, concatSt);
    resize(concatSt *255, concatSt, Size(), 10, 10, InterpolationFlags::INTER_NEAREST);
    imshow("result", concatSt);

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
        Point lt((rect.x                    ) * rate      + i, (rect.y                     ) * rate      + i);
        Point lb((rect.x                    ) * rate      + i, (rect.y + (1 << rect.height)) * rate - 15 + i);
        Point rb((rect.x + (1 << rect.width)) * rate - 15 + i, (rect.y + (1 << rect.height)) * rate - 15 + i);
        Point rt((rect.x + (1 << rect.width)) * rate - 15 + i, (rect.y                     ) * rate      + i);
        cv::line(kernel, lt, lb, color[i], 2);
        cv::line(kernel, lb, rb, color[i], 2);
        cv::line(kernel, rb, rt, color[i], 2);
        cv::line(kernel, rt, lt, color[i], 2);
    }
    imshow("kernel", kernel);

    waitKey();
    destroyAllWindows();
}

} //
} // opencv_test
