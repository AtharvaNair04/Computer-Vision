#include <math.h>
#include "mex.h"

#define round(x) ((x-floor(x))>0.5 ? ceil(x) : floor(x))
#define eps 0.0001

double uu[9] = {1.0000,
                0.9397,
                0.7660,
                0.500,
                0.1736,
                -0.1736,
                -0.5000,
                -0.7660,
                -0.9397};

double vv[9] = {0.0000,
                0.3420,
                0.6428,
                0.8660,
                0.9848,
                0.9848,
                0.8660,
                0.6428,
                0.3420};

static inline double min(double x, double y)
{
    return (x <= y ? x : y);
}

static inline double max(double x, double y)
{
    return (x <= y ? y : x);
}

static inline int min(int x, int y)
{
    return (x <= y ? x : y);
}

static inline int max(int x, int y)
{
    return (x <= y ? y : x);
}


mxArray *process(const mxArray *mximage, const mxArray *mxsbin)
{
    double *im = (double *)mxGetPr(mximage);

    const mwSize *dims = mxGetDimensions(mximage);

    if (mxGetNumberOfDimensions(mximage) != 3 ||
        dims[2] != 3 ||
        mxGetClassID(mximage) != mxDOUBLE_CLASS)
    {
        mexErrMsgTxt("Invalid input");
    }

    int sbin = (int)mxGetScalar(mxsbin);

    int height = (int)dims[0];
    int width  = (int)dims[1];

    /*
     * ------------------------------------------------------------
     * IMAGE QUALITY ANALYSIS
     * ------------------------------------------------------------
     */

    double sum = 0.0;
    double sumSq = 0.0;
    double count = 0.0;

    for (int x = 0; x < width; x++)
    {
        for (int y = 0; y < height; y++)
        {
            int idx = y + x * height;

            double r = im[idx] / 255.0;
            double g = im[idx + height * width] / 255.0;
            double b = im[idx + 2 * height * width] / 255.0;

            r = min(max(r, 0.0), 1.0);
            g = min(max(g, 0.0), 1.0);
            b = min(max(b, 0.0), 1.0);

            double gray =
                0.299 * r +
                0.587 * g +
                0.114 * b;

            sum += gray;
            sumSq += gray * gray;
            count += 1.0;
        }
    }

    double brightness = sum / count;

    double variance =
        sumSq / count -
        brightness * brightness;

    if (variance < 0.0)
        variance = 0.0;

    double contrast = sqrt(variance);


    /*
     * ------------------------------------------------------------
     * NOISE ESTIMATION
     * ------------------------------------------------------------
     */

    double noiseSum = 0.0;
    double noiseCount = 0.0;

    if (width >= 3 && height >= 3)
    {
        for (int x = 1; x < width - 1; x++)
        {
            for (int y = 1; y < height - 1; y++)
            {
                int idx = y + x * height;

                int idxLeft  = y + (x - 1) * height;
                int idxRight = y + (x + 1) * height;
                int idxUp    = (y - 1) + x * height;
                int idxDown  = (y + 1) + x * height;

                double gray =
                    0.299 * (im[idx] / 255.0) +
                    0.587 * (im[idx + height * width] / 255.0) +
                    0.114 * (im[idx + 2 * height * width] / 255.0);

                double grayLeft =
                    0.299 * (im[idxLeft] / 255.0) +
                    0.587 * (im[idxLeft + height * width] / 255.0) +
                    0.114 * (im[idxLeft + 2 * height * width] / 255.0);

                double grayRight =
                    0.299 * (im[idxRight] / 255.0) +
                    0.587 * (im[idxRight + height * width] / 255.0) +
                    0.114 * (im[idxRight + 2 * height * width] / 255.0);

                double grayUp =
                    0.299 * (im[idxUp] / 255.0) +
                    0.587 * (im[idxUp + height * width] / 255.0) +
                    0.114 * (im[idxUp + 2 * height * width] / 255.0);

                double grayDown =
                    0.299 * (im[idxDown] / 255.0) +
                    0.587 * (im[idxDown + height * width] / 255.0) +
                    0.114 * (im[idxDown + 2 * height * width] / 255.0);

                double localAverage =
                    0.25 *
                    (grayLeft +
                     grayRight +
                     grayUp +
                     grayDown);

                double residual =
                    fabs(gray - localAverage);

                noiseSum += residual;
                noiseCount += 1.0;
            }
        }
    }

    double noiseLevel = 0.0;

    if (noiseCount > 0.0)
        noiseLevel = noiseSum / noiseCount;


    /*
     * ------------------------------------------------------------
     * SHARPNESS ESTIMATION
     * ------------------------------------------------------------
     */

    double sharpnessSum = 0.0;
    double sharpnessSumSq = 0.0;
    double sharpnessCount = 0.0;

    if (width >= 3 && height >= 3)
    {
        for (int x = 1; x < width - 1; x++)
        {
            for (int y = 1; y < height - 1; y++)
            {
                int idx = y + x * height;

                int idxLeft  = y + (x - 1) * height;
                int idxRight = y + (x + 1) * height;
                int idxUp    = (y - 1) + x * height;
                int idxDown  = (y + 1) + x * height;

                double gray =
                    0.299 * (im[idx] / 255.0) +
                    0.587 * (im[idx + height * width] / 255.0) +
                    0.114 * (im[idx + 2 * height * width] / 255.0);

                double grayLeft =
                    0.299 * (im[idxLeft] / 255.0) +
                    0.587 * (im[idxLeft + height * width] / 255.0) +
                    0.114 * (im[idxLeft + 2 * height * width] / 255.0);

                double grayRight =
                    0.299 * (im[idxRight] / 255.0) +
                    0.587 * (im[idxRight + height * width] / 255.0) +
                    0.114 * (im[idxRight + 2 * height * width] / 255.0);

                double grayUp =
                    0.299 * (im[idxUp] / 255.0) +
                    0.587 * (im[idxUp + height * width] / 255.0) +
                    0.114 * (im[idxUp + 2 * height * width] / 255.0);

                double grayDown =
                    0.299 * (im[idxDown] / 255.0) +
                    0.587 * (im[idxDown + height * width] / 255.0) +
                    0.114 * (im[idxDown + 2 * height * width] / 255.0);

                double laplacian =
                    grayLeft +
                    grayRight +
                    grayUp +
                    grayDown -
                    4.0 * gray;

                sharpnessSum += laplacian;
                sharpnessSumSq += laplacian * laplacian;
                sharpnessCount += 1.0;
            }
        }
    }

    double sharpness = 0.0;

    if (sharpnessCount > 0.0)
    {
        double mean =
            sharpnessSum / sharpnessCount;

        sharpness =
            sharpnessSumSq / sharpnessCount -
            mean * mean;

        if (sharpness < 0.0)
            sharpness = 0.0;
    }


    /*
     * ------------------------------------------------------------
     * ADAPTATION FLAGS
     * ------------------------------------------------------------
     */

    bool lowContrast =
        (contrast < 0.12);

    bool highNoise =
        (noiseLevel > 0.045);

    bool blurred =
        (sharpness < 0.001);


    /*
     * ------------------------------------------------------------
     * HOG MEMORY
     * ------------------------------------------------------------
     */

    int blocks[2];

    blocks[0] =
        (int)round(
            (double)dims[0] /
            (double)sbin);

    blocks[1] =
        (int)round(
            (double)dims[1] /
            (double)sbin);

    double *hist =
        (double *)mxCalloc(
            blocks[0] *
            blocks[1] *
            18,
            sizeof(double));

    double *norm =
        (double *)mxCalloc(
            blocks[0] *
            blocks[1],
            sizeof(double));


    /*
     * ------------------------------------------------------------
     * OUTPUT: SAME 32 FMP CHANNELS
     * ------------------------------------------------------------
     */

    mwSize out[3];

    out[0] = max(blocks[0] - 2, 0);
    out[1] = max(blocks[1] - 2, 0);
    out[2] = 32;

    mxArray *mxfeat =
        mxCreateNumericArray(
            3,
            out,
            mxDOUBLE_CLASS,
            mxREAL);

    double *feat =
        (double *)mxGetPr(mxfeat);


    int visible[2];

    visible[0] =
        blocks[0] * sbin;

    visible[1] =
        blocks[1] * sbin;


    /*
     * ------------------------------------------------------------
     * GRADIENT + ADAPTIVE WEIGHTING
     * ------------------------------------------------------------
     */

    for (int x = 1;
         x < visible[1] - 1;
         x++)
    {
        for (int y = 1;
             y < visible[0] - 1;
             y++)
        {
            /*
             * RED
             */
            double *s =
                im +
                min(x, dims[1] - 2) * dims[0] +
                min(y, dims[0] - 2);

            double dy =
                *(s + 1) -
                *(s - 1);

            double dx =
                *(s + dims[0]) -
                *(s - dims[0]);

            double v =
                dx * dx +
                dy * dy;


            /*
             * GREEN
             */
            s += dims[0] * dims[1];

            double dy2 =
                *(s + 1) -
                *(s - 1);

            double dx2 =
                *(s + dims[0]) -
                *(s - dims[0]);

            double v2 =
                dx2 * dx2 +
                dy2 * dy2;


            /*
             * BLUE
             */
            s += dims[0] * dims[1];

            double dy3 =
                *(s + 1) -
                *(s - 1);

            double dx3 =
                *(s + dims[0]) -
                *(s - dims[0]);

            double v3 =
                dx3 * dx3 +
                dy3 * dy3;


            /*
             * Strongest RGB channel
             */
            if (v2 > v)
            {
                v = v2;
                dx = dx2;
                dy = dy2;
            }

            if (v3 > v)
            {
                v = v3;
                dx = dx3;
                dy = dy3;
            }


            /*
             * Gradient magnitude
             */
            v = sqrt(v);

            double originalMagnitude = v;


            /*
             * LOW CONTRAST
             *
             * Boost weak gradients more than
             * strong gradients.
             */
            if (lowContrast)
            {
                if (originalMagnitude < 32.0)
                {
                    double g =
                        originalMagnitude / 32.0;

                    if (g > 1.0)
                        g = 1.0;

                    double boost =
                        1.0 +
                        0.10 * (1.0 - g);

                    v *= boost;
                }
            }


            /*
             * HIGH NOISE
             *
             * Suppress weak gradients.
             */
            if (highNoise)
            {
                if (originalMagnitude < 12.0)
                {
                    v *= 0.70;
                }
                else if (originalMagnitude < 24.0)
                {
                    v *= 0.85;
                }
            }


            /*
             * BLUR
             *
             * Avoid amplifying weak gradients.
             */
            if (blurred)
            {
                if (originalMagnitude < 10.0)
                {
                    v *= 0.85;
                }
            }


            /*
             * Keep magnitude bounded.
             */
            if (v > 255.0)
                v = 255.0;


            /*
             * ------------------------------------------------
             * ORIENTATION
             * ------------------------------------------------
             */

            double best_dot = 0;
            int best_o = 0;

            for (int o = 0; o < 9; o++)
            {
                double dot =
                    uu[o] * dx +
                    vv[o] * dy;

                if (dot > best_dot)
                {
                    best_dot = dot;
                    best_o = o;
                }
                else if (-dot > best_dot)
                {
                    best_dot = -dot;
                    best_o = o + 9;
                }
            }


            /*
             * ------------------------------------------------
             * SPATIAL INTERPOLATION
             * ------------------------------------------------
             */

            double xp =
                ((double)x + 0.5) /
                (double)sbin -
                0.5;

            double yp =
                ((double)y + 0.5) /
                (double)sbin -
                0.5;

            int ixp =
                (int)floor(xp);

            int iyp =
                (int)floor(yp);

            double vx0 =
                xp - ixp;

            double vy0 =
                yp - iyp;

            double vx1 =
                1.0 - vx0;

            double vy1 =
                1.0 - vy0;


            if (ixp >= 0 &&
                iyp >= 0)
            {
                *(hist +
                  ixp * blocks[0] +
                  iyp +
                  best_o *
                  blocks[0] *
                  blocks[1]) +=
                    vx1 * vy1 * v;
            }

            if (ixp + 1 < blocks[1] &&
                iyp >= 0)
            {
                *(hist +
                  (ixp + 1) *
                  blocks[0] +
                  iyp +
                  best_o *
                  blocks[0] *
                  blocks[1]) +=
                    vx0 * vy1 * v;
            }

            if (ixp >= 0 &&
                iyp + 1 < blocks[0])
            {
                *(hist +
                  ixp * blocks[0] +
                  (iyp + 1) +
                  best_o *
                  blocks[0] *
                  blocks[1]) +=
                    vx1 * vy0 * v;
            }

            if (ixp + 1 < blocks[1] &&
                iyp + 1 < blocks[0])
            {
                *(hist +
                  (ixp + 1) *
                  blocks[0] +
                  (iyp + 1) +
                  best_o *
                  blocks[0] *
                  blocks[1]) +=
                    vx0 * vy0 * v;
            }
        }
    }


    /*
     * ------------------------------------------------------------
     * NORMALIZATION
     * ------------------------------------------------------------
     */

    for (int o = 0; o < 9; o++)
    {
        double *src1 =
            hist +
            o *
            blocks[0] *
            blocks[1];

        double *src2 =
            hist +
            (o + 9) *
            blocks[0] *
            blocks[1];

        double *dst = norm;

        double *end =
            norm +
            blocks[1] *
            blocks[0];

        while (dst < end)
        {
            *(dst++) +=
                (*src1 + *src2) *
                (*src1 + *src2);

            src1++;
            src2++;
        }
    }


    /*
     * ------------------------------------------------------------
     * BUILD 32 FEATURES
     * ------------------------------------------------------------
     */

    for (int x = 0;
         x < out[1];
         x++)
    {
        for (int y = 0;
             y < out[0];
             y++)
        {
            double *dst =
                feat +
                x * out[0] +
                y;

            double *src;

            double n1, n2, n3, n4;


            /*
             * Normalization 1
             */
            double *p =
                norm +
                (x + 1) *
                blocks[0] +
                y + 1;

            n1 =
                1.0 /
                sqrt(
                    *p +
                    *(p + 1) +
                    *(p + blocks[0]) +
                    *(p + blocks[0] + 1) +
                    eps);


            /*
             * Normalization 2
             */
            p =
                norm +
                (x + 1) *
                blocks[0] +
                y;

            n2 =
                1.0 /
                sqrt(
                    *p +
                    *(p + 1) +
                    *(p + blocks[0]) +
                    *(p + blocks[0] + 1) +
                    eps);


            /*
             * Normalization 3
             */
            p =
                norm +
                x *
                blocks[0] +
                y + 1;

            n3 =
                1.0 /
                sqrt(
                    *p +
                    *(p + 1) +
                    *(p + blocks[0]) +
                    *(p + blocks[0] + 1) +
                    eps);


            /*
             * Normalization 4
             */
            p =
                norm +
                x *
                blocks[0] +
                y;

            n4 =
                1.0 /
                sqrt(
                    *p +
                    *(p + 1) +
                    *(p + blocks[0]) +
                    *(p + blocks[0] + 1) +
                    eps);


            double t1 = 0;
            double t2 = 0;
            double t3 = 0;
            double t4 = 0;


            /*
             * 18 contrast-sensitive features
             */
            src =
                hist +
                (x + 1) *
                blocks[0] +
                (y + 1);

            for (int o = 0;
                 o < 18;
                 o++)
            {
                double h1 =
                    min(*src * n1, 0.2);

                double h2 =
                    min(*src * n2, 0.2);

                double h3 =
                    min(*src * n3, 0.2);

                double h4 =
                    min(*src * n4, 0.2);

                *dst =
                    0.5 *
                    (h1 + h2 + h3 + h4);

                t1 += h1;
                t2 += h2;
                t3 += h3;
                t4 += h4;

                dst +=
                    out[0] *
                    out[1];

                src +=
                    blocks[0] *
                    blocks[1];
            }


            /*
             * 9 contrast-insensitive features
             */
            src =
                hist +
                (x + 1) *
                blocks[0] +
                (y + 1);

            for (int o = 0;
                 o < 9;
                 o++)
            {
                double sum =
                    *src +
                    *(src +
                      9 *
                      blocks[0] *
                      blocks[1]);

                double h1 =
                    min(sum * n1, 0.2);

                double h2 =
                    min(sum * n2, 0.2);

                double h3 =
                    min(sum * n3, 0.2);

                double h4 =
                    min(sum * n4, 0.2);

                *dst =
                    0.5 *
                    (h1 + h2 + h3 + h4);

                dst +=
                    out[0] *
                    out[1];

                src +=
                    blocks[0] *
                    blocks[1];
            }


            /*
             * 4 texture features
             */
            *dst =
                0.2357 * t1;

            dst +=
                out[0] *
                out[1];

            *dst =
                0.2357 * t2;

            dst +=
                out[0] *
                out[1];

            *dst =
                0.2357 * t3;

            dst +=
                out[0] *
                out[1];

            *dst =
                0.2357 * t4;


            /*
             * Truncation feature
             */
            dst +=
                out[0] *
                out[1];

            *dst = 0;
        }
    }


    mxFree(hist);
    mxFree(norm);

    return mxfeat;
}


/*
 * MATLAB entry point
 */
void mexFunction(
    int nlhs,
    mxArray *plhs[],
    int nrhs,
    const mxArray *prhs[])
{
    if (nrhs != 2)
        mexErrMsgTxt("Wrong number of inputs");

    if (nlhs != 1)
        mexErrMsgTxt("Wrong number of outputs");

    plhs[0] =
        process(
            prhs[0],
            prhs[1]);
}