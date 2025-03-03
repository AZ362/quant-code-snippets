import numpy as np

list_patterns = ["gartley_range", "butterfly", "bat", "altbat", "crab", "shark", "cypher", "abcd"]

def is_gartley_range(moves, err_allowed):
    XA = moves[0]
    AB = moves[1]
    BC = moves[2]
    CD = moves[3]
    AD = moves[4]
    retVal = np.NAN

    AB_range = np.array([0.618 - err_allowed, 0.618 + err_allowed]) * abs(XA)
    BC_range = np.array([0.382 - err_allowed, 0.886 + err_allowed]) * abs(AB)
    CD_range = np.array([1.13 - err_allowed, 1.618 + err_allowed]) * abs(AB)
    AD_range = np.array([0.786 - err_allowed, 0.786 + err_allowed]) * abs(XA)
    ABAD_range = np.array([1 - err_allowed, 1 + err_allowed]) * abs(AB)
    if XA > 0 and AB < 0 and BC > 0 and CD < 0:  # and AD < 0:
        if AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1] and CD_range[0] < abs(CD) < \
                CD_range[1]:
            # and AD_range[0] < abs(AD) <AD_range[1]):
            retVal = 1
    elif XA < 0 and AB > 0 and BC < 0 and CD > 0:  # and AD > 0 :
        if AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1] and CD_range[0] < abs(CD) < \
                CD_range[1]:
            # and AD_range[0] < abs(AD) < AD_range[1]):
            retVal = -1

    return retVal


def is_butterfly(moves, err_allowed):
    XA = moves[0]
    AB = moves[1]
    BC = moves[2]
    CD = moves[3]
    AD = moves[4]
    retVal = np.NAN

    AB_range = np.array([0.786 - err_allowed, 0.786 + err_allowed]) * abs(XA)
    BC_range = np.array([0.382 - err_allowed, 0.886 + err_allowed]) * abs(AB)
    CD_range = np.array([1.618 - err_allowed, 2.24 + err_allowed]) * abs(BC)
    AD_range = np.array([1.272 - err_allowed, 1.272 + err_allowed]) * abs(XA)

    if XA > 0 and AB < 0 and BC > 0 and CD < 0 and AD < 0:
        if AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1] and CD_range[0] < abs(CD) < \
                CD_range[1] and AD_range[0] < abs(AD) < AD_range[1]:
            retVal = 1
    elif XA < 0 and AB > 0 and BC < 0 and CD > 0 and AD > 0:
        if AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1] and CD_range[0] < abs(CD) < \
                CD_range[1] and AD_range[0] < abs(AD) < AD_range[1]:
            retVal = -1

    return retVal

def is_bat(moves, err_allowed):
    XA = moves[0]
    AB = moves[1]
    BC = moves[2]
    CD = moves[3]
    AD = moves[4]

    retVal = np.NAN

    AB_range = np.array([0.382 - err_allowed, 0.5 + err_allowed]) * abs(XA)  # Leg1
    BC_range = np.array([0.382 - err_allowed, 0.886 + err_allowed]) * abs(AB)  # leg2
    CD_range = np.array([1.618 - err_allowed, 2.618 + err_allowed]) * abs(BC)  # leg3
    AD_range = np.array([0.886 - err_allowed, 0.886 + err_allowed]) * abs(XA)  # leg4

    if XA > 0 and AB < 0 and BC > 0 and CD < 0 and AD < 0:
        if AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1] and CD_range[0] < abs(CD) < \
                CD_range[1] and AD_range[0] < abs(AD) < AD_range[1]:
            retVal = 1
    elif XA < 0 and AB > 0 and BC < 0 and CD > 0 and AD > 0:
        if AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1] and CD_range[0] < abs(CD) < \
                CD_range[1] and AD_range[0] < abs(AD) < AD_range[1]:
            retVal = -1

    return retVal

def is_altbat(moves, err_allowed):
    XA = moves[0]
    AB = moves[1]
    BC = moves[2]
    CD = moves[3]
    AD = moves[4]

    retVal = np.NAN

    AB_range = np.array([0.382 - err_allowed, 0.382 + err_allowed]) * abs(XA)  # Leg1
    BC_range = np.array([0.382 - err_allowed, 0.886 + err_allowed]) * abs(AB)  # leg2
    CD_range = np.array([2 - err_allowed, 3.168 + err_allowed]) * abs(BC)  # leg3
    AD_range = np.array([1.13 - err_allowed, 1.13 + err_allowed]) * abs(XA)  # leg4

    if XA > 0 and AB < 0 and BC > 0 and CD < 0 and AD < 0:
        if AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1] and CD_range[0] < abs(CD) < \
                CD_range[1] and AD_range[0] < abs(AD) < AD_range[1]:
            retVal = 1
    elif XA < 0 and AB > 0 and BC < 0 and CD > 0 and AD > 0:
        if AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1] and CD_range[0] < abs(CD) < \
                CD_range[1] and AD_range[0] < abs(AD) < AD_range[1]:
            retVal = -1

    return retVal

def is_crab(moves, err_allowed):#= 5.0/100
    XA = moves[0]
    AB = moves[1]
    BC = moves[2]
    CD = moves[3]
    AD = moves[4]

    retVal = np.NAN

    AB_range = np.array([0.382 - err_allowed, 0.618 + err_allowed]) * abs(XA)
    BC_range = np.array([0.382 - err_allowed, 0.886 + err_allowed]) * abs(AB)
    CD_range = np.array([2.618 - err_allowed, 3.618 + err_allowed]) * abs(BC)
    AD_range = np.array([1.618 - err_allowed*2, 1.618 + err_allowed*2]) * abs(XA)

    if XA > 0 and AB < 0 and BC > 0 and CD < 0 and AD < 0:
        if AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1] and CD_range[0] < abs(CD) < \
                CD_range[1]:# and AD_range[0] < abs(AD) < AD_range[1]:
            retVal = 1
    elif XA < 0 and AB > 0 and BC < 0 and CD > 0 and AD > 0:
        if AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1] and CD_range[0] < abs(CD) < \
                CD_range[1]:# and AD_range[0] < abs(AD) < AD_range[1]:
            retVal = -1

    return retVal

def is_shark(moves, err_allowed):
    XA = moves[0]
    AB = moves[1]
    BC = moves[2]
    CD = moves[3]
    AD = moves[4]
    XC= moves[0]+moves[1]+ moves[2]
    retVal = np.NAN

    # AB_range = np.array([0.5 - err_allowed,0.886 + err_allowed]) * abs(XA)
    BC_range = np.array([1.13 - err_allowed, 1.618 + err_allowed]) * abs(AB)
    CD_range = np.array([1.618 - err_allowed, 2.24 + err_allowed]) * abs(BC)
    XCD_range = np.array([0.886 - err_allowed, 1.13 + err_allowed]) * abs(XC)
    if XA > 0 and AB < 0 and BC > 0 and CD < 0 and AD < 0:
        if BC_range[0] < abs(BC) < BC_range[1] and CD_range[0] < abs(CD) < CD_range[1] and XCD_range[0] < abs(CD) < XCD_range[1]:
            retVal = 1
    elif XA < 0 and AB > 0 and BC < 0 and CD > 0 and AD > 0:
        if BC_range[0] < abs(BC) < BC_range[1] and CD_range[0] < abs(CD) < CD_range[1] and XCD_range[0] < abs(CD) < XCD_range[1]:
            retVal = -1

    return retVal

def is_cypher(moves, err_allowed):## brst error: 10%>>
    XA = moves[0]
    AB = moves[1]
    BC = moves[2]
    CD = moves[3]
    #AD = moves[4]
    XC= moves[0]+moves[1]+ moves[2]
    retVal = np.NAN
    AB_range = np.array([0.382 - err_allowed, 0.618 + err_allowed]) * abs(XA)
    BC_range = np.array([1.272 - err_allowed, 1.414 + err_allowed]) * abs(AB)
    CD_range = np.array([1.272 - err_allowed, 2.0 + err_allowed]) * abs(BC)
    #XCD_range = np.array([0.786 - err_allowed, 0.786 + err_allowed]) * abs(XC)

    if XA > 0 and AB < 0 and BC > 0 and CD < 0:
        if AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1] and CD_range[0] < abs(CD) < \
                CD_range[1]:# and XCD_range[0] < abs(CD) < XCD_range[1]:
            retVal = 1
    elif XA < 0 and AB > 0 and BC < 0 and CD > 0:
        if AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1] and CD_range[0] < abs(CD) < \
                CD_range[1]:# and XCD_range[0] < abs(CD) < XCD_range[1]:
            retVal = -1

    return retVal


# B-C leg is 61.8% retracement of A-B leg
# C-D leg is the 127.2% extension of B-C leg

def is_abcd(moves, err_allowed):
    XA = moves[0]
    AB = moves[1]
    BC = moves[2]
    CD = moves[3]
    AD = moves[4]
    retVal = np.NAN
    AB_range = np.array([1 - err_allowed, 1 + err_allowed]) * abs(CD)
    BC_range = np.array([0.618 - err_allowed, 0.618 + err_allowed]) * abs(AB)
    CD_range = np.array([1.13 - err_allowed, 1.27 + err_allowed]) * abs(BC)
    # AD_range = np.array([0.786 - err_allowed,0.786 + err_allowed]) * abs(XA)

    if AB < 0 and BC > 0 and CD < 0:
        if AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1] and CD_range[0] < abs(CD) < \
                CD_range[1]:
            retVal = 1
    elif AB > 0 and BC < 0 and CD > 0:
        if AB_range[0] < abs(AB) < AB_range[1] and BC_range[0] < abs(BC) < BC_range[1] and CD_range[0] < abs(CD) < \
                CD_range[1]:
            retVal = -1

    return retVal