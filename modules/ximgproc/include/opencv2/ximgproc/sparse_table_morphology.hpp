// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef __OPENCV_SPARSE_TABLE_MORPHOLOGY_HPP__
#define __OPENCV_SPARSE_TABLE_MORPHOLOGY_HPP__

#include <opencv2/core.hpp>
#include <vector>

namespace cv {
namespace ximgproc {
namespace stMorph {

//! @addtogroup imgproc_filter
//! @{

/**
 * @brief Faster implementation of cv::erode with sparse table concept.
 *
 * @param src input image; the number of channels can be arbitrary, but the depth should be one of
 * CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
 * @param dst output image of the same size and type as src.
 * @param kernel structuring element used for erosion; if `element=Mat()`, a `3 x 3` rectangular
 * structuring element is used. Kernel can be created using #getStructuringElement.
 * @param anchor position of the anchor within the element; default value (-1, -1) means that the
 * anchor is at the element center.
 * @param borderType pixel extrapolation method, see #BorderTypes. #BORDER_WRAP is not supported.
 * @param borderValue border value in case of a constant border
 *
 * @see cv::erode
 */
CV_EXPORTS_W void erode( InputArray src, OutputArray dst, InputArray kernel, Point anchor = Point(-1,-1),
                          int borderType = BORDER_CONSTANT, const Scalar& borderValue = morphologyDefaultBorderValue() );

/**
 * @brief Faster implementation of cv::dilate with sparse table concept.
 *
 * @param src input image; the number of channels can be arbitrary, but the depth should be one of
 * CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
 * @param dst output image of the same size and type as src.
 * @param kernel structuring element used for dilation; if element=Mat(), a 3 x 3 rectangular
 * structuring element is used. Kernel can be created using #getStructuringElement
 * @param anchor position of the anchor within the element; default value (-1, -1) means that the
 * anchor is at the element center.
 * @param borderType pixel extrapolation method, see #BorderTypes. #BORDER_WRAP is not suported.
 * @param borderValue border value in case of a constant border
 *
 * @see cv::dilate
 */
CV_EXPORTS_W void dilate( InputArray src, OutputArray dst, InputArray kernel, Point anchor = Point(-1, -1),
                          int borderType = BORDER_CONSTANT, const Scalar& borderValue = morphologyDefaultBorderValue() );

/**
 * @brief Faster implementation of cv::morphologyEx with sparse table concept.

 * @param src Source image. The number of channels can be arbitrary. The depth should be one of
 * CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
 * @param dst Destination image of the same size and type as source image.
 * @param op Type of a morphological operation, see #MorphTypes
 * @param kernel Structuring element. It can be created using #getStructuringElement.
 * @param anchor Anchor position with the kernel. Negative values mean that the anchor is at the
 * kernel center.
 * @param iterations Number of times erosion and dilation are applied.
 * @param borderType Pixel extrapolation method, see #BorderTypes. #BORDER_WRAP is not supported.
 * @param borderValue Border value in case of a constant border. The default value has a special
 * meaning.
 * @note The number of iterations is the number of times erosion or dilatation operation will be applied.
 * For instance, an opening operation (#MORPH_OPEN) with two iterations is equivalent to apply
 * successively: erode -> erode -> dilate -> dilate (and not erode -> dilate -> erode -> dilate).
 *
 * @see cv::morphologyEx
 */
CV_EXPORTS_W void morphologyEx( InputArray src, OutputArray dst, int op, InputArray kernel, Point anchor = Point(-1,-1),
                                int iterations = 1,
                                int borderType = BORDER_CONSTANT, const Scalar& borderValue = morphologyDefaultBorderValue() );

//! @}

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

enum Dim
{
    Col, Row
};

struct StStep
{
    StStep(int dimR, int dimC, Dim _ax)
    {
        dimRow = dimR;
        dimCol = dimC;
        ax = _ax;
    }
    int dimRow;
    int dimCol;
    Dim ax;
};

/*
* Find a smaller set of power-of-2 rectangles to cover the kernel.
* - The width and the height of each rectangles are power of 2.
* - Overlappings of rectangles are allowed.
*
* this method may be applied for the covering polygon problem with rectangle.
* https://www.sciencedirect.com/science/article/pii/S0019995884800121z
*/
CV_EXPORTS_W std::vector<Rect> genPow2RectsToCoverKernel(InputArray kernel);
//
///*
//* Plan the order to calculate the sparse table nodes.
//*/
CV_EXPORTS_W std::vector<StStep> planSparseTableConstruction(std::vector<std::vector<bool>> requiredSparseTableNodeMap);

} // namespace st
} // namespace ximgproc
} // namespace cv

#endif
