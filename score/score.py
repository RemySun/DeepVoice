#!/usr/bin/python

import sys
import getopt
import re
import math
# import trstk

# ----------------------- #
# ----- def usage() ----- #
# ----------------------- #
def usage():
	print "Usage: " +  sys.argv[0] + " [options] file.score"
        print "  -o, --output=fn      set ROC data output filename"
	print "  -h, --help           print help"
        sys.exit(1)

# print 'Argument is ', sys.argv[1]

# ----------------------- #
# ----- def stats() ----- #
# ----------------------- #
def stats(a):
    "compute mean and variance for values in a"
    n = len(a)
    m = 0
    v = 0

    for val in a:
        m += val
        v += (val * val)
    
    m /= n

    return n, m, (v / n) - m * m

# --------------------- #
# ----- def kl2() ----- #
# --------------------- #
def kl2(m1,v1,m2,v2):
    "compute KL2 divergence between two Gaussian distributions"
    dif = (m2 - m1) * (m2 - m1)
    d1 = 0.5 * (math.log(v2 / v1) + v1 / v2 + dif / v2 - 1)
    d2 = 0.5 * (math.log(v1 / v2) + v2 / v1 + dif / v1 - 1)

    return 0.5 * (d1 + d2) / math.log(2)

# ----------------------- #
# ----- def ppndf() ----- #
# ----------------------- #
def ppndf(p):
    eps = 2.2204e-16
    
    p = float(p)

    if (p >= 1.0):
        p = p - eps
    elif (p <= 0.0):
        p = eps

    q = p - 0.5
    if math.fabs(q) <= 0.42:
        r = q * q
        retval = q * (((-25.4410604963 * r + 41.3911977353) * r - 18.6150006252) * r + 2.5066282388) / ((((3.1308290983 * r - 21.0622410182) * r + 23.0833674374) * r - 8.4735109309) *r + 1.0)

    else:
        if q > 0.0:
            r = 1.0 - p
        else:
            r = p

        assert (r > 0), "bad value %f for r in ppndf()" % (r)

        r = math.sqrt(-1.0 * math.log(r))

        retval = (((2.3212127685 * r + 4.8501412713) * r - 2.2979647913) * r - 2.7871893113) / ((1.6370678189 * r + 3.5438892476) *r + 1.0)

        if q < 0.0: retval *= -1.0

    return retval

# --------------------------- #
# ----- def rocpoints() ----- #
# --------------------------- #
def rocpoints(true_score,false_score):
    "compute ROC points from raw scores for DET plotting"
    pts = []   # each entry is a triplet threshold, FA, FR

    delta_error = 100;

    # sort both lists of scores in ascending order
    true_score.sort()
    false_score.sort()

    ntrue = len(true_score)
    nfalse = len(false_score)

    i = 0
    j = 0

    x1 = true_score[0]
    x2 = false_score[0]

    while i < ntrue and j < nfalse: # scan all values in the two lists
        ii = i
        jj = j

        if true_score[i] <= false_score[j]: # take true_score[i] as base, deeming same speaker all trials whose score is less than true_score[i] 
            th = true_score[i]
            ii = i + 1

        if true_score[i] >= false_score[j]: # take false_score[j] as base, deeming same speaker all trials whose score is less than false_score[j] 
            th = false_score[j]
            jj = j + 1

        fr = float(1) - float(ii) / float(ntrue)
        fa = float(jj) / float(nfalse)
        
        if fr < 0.5 and fa < 0.5:
            pts.append([th, fr, fa])
            # print "i=%d  j=%d  true=%.4f false=%.4f th=%.4f fr=%.2f fa=%.2f" % (i, j, true_score[i], false_score[j], th, 100 * fr, 100 * fa)
        
        i = ii
        j = jj

        # check for EER update
        dif = math.fabs(fr - fa)
        if dif < delta_error:
            delta_error = dif
            eer_fr = fr
            eer_fa = fa
    # end while  i < ntrue and j < nfalse:

    # print 'equal error rate: ', delta_error, eer_fr, eer_fa
    eer = 0.5 * (eer_fa + eer_fr)

    return eer, pts
 
# ------------------ #
# ----- main() ----- #
# ------------------ #
if __name__ == "__main__":

    # print 'C parti mon pote !'

    # ======================================================================
    #
    # Process command line options (if any)
    #
    try:
        opts, args = getopt.getopt(sys.argv[1:], "ho:", ["help", "output="])
    except getopt.GetoptError as err:
        # print help information and exit:
        print(err) # will print something like "option -a not recognized"
        sys.exit(-1)

    outfn = ''
    for o, a in opts:
        if o in ("-h", "--help"):
            usage()
            sys.exit()
        elif o in ("-o", "--output"):
                outfn = a
        else:
            assert False, "unhandled option"
            sys.exit(-1)


    # ======================================================================
    #
    # Check/Process command line arguments
    #
    if len(args) != 1:
        usage();

    # ======================================================================
    #
    # Loop on all records in input file
    #
    infile = open(args[0], "r")
    i = 0

    #
    # the score array contains all scores broken down by trial types
    # score[0] = diff speaker pairs / score[1] = same speaker pairs
    # score[.][0] = diff sex, diff show     // cannot happen for same speaker pairs
    # score[.][1] = same sex, diff show
    # score[.][2] = diff sex, same show     // cannot happen for same speaker pairs
    # score[.][3] = same sex, same show
    #
    score = [[[], [], [], []], [[], [], [], []]] # initialize 2d array of lists

    for line in infile:
        line = line.rstrip("\n")

        # line format: seg1 seg2 same_speaker_flag same_gender_flag same_show_flag score
        res = re.match( r'^(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)$', line, re.M)

        # sys.stderr.write("===== score.py debug ===== %06d %s    > " % (i, line))
        if res:
            tab = res.groups()
                
            index = int(tab[3]) + 2 * int(tab[4])
            #print tab, index
            # sys.stderr.write(" %s %s class %s sex %s show %s score %s    -- index[%d]=%d\n" % (tab[0], tab[1], tab[2], tab[3], tab[4], tab[5], int(tab[2]), index))

            score[int(tab[2])][index].append(float(tab[5]))
        else:
            sys.stderr.write('===== score.py error ===== no fucking match - ignoring line\n' % line)
 
        i = i + 1
    
    infile.close()

    # ======================================================================
    #
    # Compute statistics for scores
    #
    print 'same speaker pairs'

    n13,m13,v13 = stats(score[1][3])
    print "   same show: %6d %7.4f %6.3f   ]%.3f,%.3f[" % (n13, m13, v13, m13 - 1.64 * math.sqrt(v13), m13 + 1.64 * math.sqrt(v13))
    n11,m11,v11 = stats(score[1][1])
    print "   diff show: %6d %7.4f %6.3f   ]%.3f,%.3f[" % (n11, m11, v11, m11 - 1.64 * math.sqrt(v11), m11 + 1.64 * math.sqrt(v11))
    n1,m1,v1 = stats(score[1][3] + score[1][1])
    print "   overall:   %6d %7.4f %6.3f   ]%.3f,%.3f[" % (n1, m1, v1, m1 - 1.64 * math.sqrt(v1), m1 + 1.64 * math.sqrt(v1))

    print 'diff speaker pairs'

    n03,m03,v03 = stats(score[0][3])
    print "   same sex, same show: %6d %7.4f %6.3f   ]%.3f,%.3f[" % (n03, m03, v03, m03 - 1.64 * math.sqrt(v03), m03 + 1.64 * math.sqrt(v03))
    n01,m01,v01 = stats(score[0][1])
    print "   same sex, diff show: %6d %7.4f %6.3f   ]%.3f,%.3f[" % (n01, m01, v01, m01 - 1.64 * math.sqrt(v01), m01 + 1.64 * math.sqrt(v01))
    n013,m013,v013 = stats(score[0][1] + score[0][3]) # same sex all shows
    print "   same sex overall:    %6d %7.4f %6.3f   ]%.3f,%.3f[" % (n013, m013, v013, m013 - 1.64 * math.sqrt(v013), m013 + 1.64 * math.sqrt(v013))
    
    n02,m02,v02 = stats(score[0][2])
    print "   diff sex, same show: %6d %7.4f %6.3f   ]%.3f,%.3f[" % (n02, m02, v02, m02 - 1.64 * math.sqrt(v02), m02 + 1.64 * math.sqrt(v02))
    n00,m00,v00 = stats(score[0][0])
    print "   diff sex, diff show: %6d %7.4f %6.3f   ]%.3f,%.3f[" % (n00, m00, v00, m00 - 1.64 * math.sqrt(v00), m00 + 1.64 * math.sqrt(v00))
    n002,m002,v002 = stats(score[0][0] + score[0][2]) # diff sex all shows
    print "   diff sex overall:    %6d %7.4f %6.3f   ]%.3f,%.3f[" % (n002, m002, v002, m002 - 1.64 * math.sqrt(v002), m002 + 1.64 * math.sqrt(v002))

    n0,m0,v0 = stats(score[0][0] + score[0][2] + score[0][1] + score[0][3])
    print "   overall:             %6d %7.4f %6.3f   ]%.3f,%.3f[" % (n0, m0, v0, m0 - 1.64 * math.sqrt(v0), m0 + 1.64 * math.sqrt(v0))

    n032,m032,v032 = stats(score[0][3] + score[0][2]) # same show all sex
    n010,m010,v010 = stats(score[0][1] + score[0][0]) # diff show all sex

    print "KL2 divergence same/diff/all sex:  %.4f  %.4f  %.4f         (all wrt global distribution of same speaker pairs)" % (kl2(m1,v1,m013,v013), kl2(m1,v1,m002,v002), kl2(m0,v0,m1,v1))
    print "KL2 divergence same/diff/all show: %.4f  %.4f  %.4f         (all wrt global distribution of same speaker pairs)" % (kl2(m1,v1,m032,v032), kl2(m1,v1,m010,v010), kl2(m0,v0,m1,v1))
    print "KL2 divergence same/diff/all show: %.4f  %.4f  %.4f         (separating same/diff speaker trials per show)" % (kl2(m13,v13,m032,v032), kl2(m11,v11,m010,v010), kl2(m0,v0,m1,v1))

    # ======================================================================
    #
    # make numbers for DET curves
    #
    eer, pts = rocpoints(score[1][3] + score[1][1], score[0][0] + score[0][2] + score[0][1] + score[0][3])
    print "equal error rate is %.2f" % (100 * eer)

    # ======================================================================
    #
    # write output file if defined
    #
    if outfn != '':
        print "output ROC data file is", outfn

        with open(outfn, "w") as f:
                for data in pts:
                        f.write("%.4f %.4f %.4f\n" % (data[0], data[1], data[2]))
        
        f.close()


    # print 'C fini mon pote !'

