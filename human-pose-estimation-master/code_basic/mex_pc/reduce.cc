#include <math.h>
#include <assert.h>
#include <string.h>
#include "mex.h"

#define round(x) ((x-floor(x)) > 0.5 ? ceil(x) : floor(x))

// reduce(im) resizes im to half its size, using a 5-tap
// binomial filter for anti-aliasing
// (see Burt & Adelson's Laplacian Pyramid paper)

// reduce each column
// result is transposed, so we can apply it twice
// for a complete reduction
void reduce1dtran(
    double *src,
    int sheight,
    double *dst,
    int dheight,
    int width,
    int chan
) {
    // Resize each column of each color channel
    memset(dst, 0, chan * width * dheight * sizeof(double));

    int y;
    double *s;
    double *d;

    for (int c = 0; c < chan; c++) {

        for (int x = 0; x < width; x++) {

            s = src + c * width * sheight + x * sheight;
            d = dst + c * dheight * width + x;

            // First row
            *d = s[0] * 0.6875
               + s[1] * 0.2500
               + s[2] * 0.0625;

            // Middle rows
            for (y = 1; y < dheight - 2; y++) {

                s += 2;
                d += width;

                *d = s[-2] * 0.0625
                   + s[-1] * 0.25
                   + s[0]  * 0.375
                   + s[1]  * 0.25
                   + s[2]  * 0.0625;
            }

            // Last two rows
            s += 2;
            d += width;

            if (dheight * 2 <= sheight) {

                *d = s[-2] * 0.0625
                   + s[-1] * 0.25
                   + s[0]  * 0.375
                   + s[1]  * 0.25
                   + s[2]  * 0.0625;

            } else {

                *d = s[1]  * 0.3125
                   + s[0]  * 0.3750
                   + s[-1] * 0.2500
                   + s[-2] * 0.0625;
            }

            s += 2;
            d += width;

            *d = s[0]    * 0.6875
               + s[-1]   * 0.2500
               + s[-2]   * 0.0625;
        }
    }
}


// Main function
// Takes a double color image
// Returns image reduced to half size
mxArray *reduce(const mxArray *mxsrc) {

    double *src = (double *)mxGetPr(mxsrc);

    /*
     * MATLAB R2026a compatibility:
     *
     * mxGetDimensions() returns mwSize*
     * rather than int*.
     */
    const mwSize *sdims = mxGetDimensions(mxsrc);

    if (mxGetNumberOfDimensions(mxsrc) != 3 ||
        mxGetClassID(mxsrc) != mxDOUBLE_CLASS) {

        mexErrMsgTxt("Invalid input");
    }

    /*
     * MATLAB R2026a compatibility:
     *
     * Dimensions passed to mxCreateNumericArray()
     * must use mwSize.
     */
    mwSize ddims[3];

    ddims[0] = (mwSize)round(sdims[0] * 0.5);
    ddims[1] = (mwSize)round(sdims[1] * 0.5);
    ddims[2] = sdims[2];

    mxArray *mxdst =
        mxCreateNumericArray(
            3,
            ddims,
            mxDOUBLE_CLASS,
            mxREAL
        );

    double *dst = (double *)mxGetPr(mxdst);

    /*
     * Temporary buffer.
     */
    mwSize tmpSize =
        ddims[0] * sdims[1] * sdims[2];

    double *tmp =
        (double *)mxCalloc(
            tmpSize,
            sizeof(double)
        );

    /*
     * First pass:
     * reduce along columns.
     */
    reduce1dtran(
        src,
        (int)sdims[0],
        tmp,
        (int)ddims[0],
        (int)sdims[1],
        (int)sdims[2]
    );

    /*
     * Second pass:
     * reduce the transposed result.
     */
    reduce1dtran(
        tmp,
        (int)sdims[1],
        dst,
        (int)ddims[1],
        (int)ddims[0],
        (int)sdims[2]
    );

    mxFree(tmp);

    return mxdst;
}


// MATLAB entry point
//
// dst = reduce(src)
//
// Image should be a color image with
// double-valued pixels.
void mexFunction(
    int nlhs,
    mxArray *plhs[],
    int nrhs,
    const mxArray *prhs[]
) {

    if (nrhs != 1) {
        mexErrMsgTxt("Wrong number of inputs");
    }

    if (nlhs != 1) {
        mexErrMsgTxt("Wrong number of outputs");
    }

    plhs[0] = reduce(prhs[0]);
}