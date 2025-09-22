import numpy as np

def P(d_e, d_q, gamma):
    """
    Theorem 1: The expected query segment accuracy score for a given event duration d_e, query duration d_q, and gamma.
    """
    if d_q >= gamma*d_e:
        return d_e*(-2*d_e*gamma**2 + 2*d_q*gamma + d_q)/(d_q*(d_e + d_q))
    else:
        return d_q / (d_e + d_q)

def Q_max(d_e, gamma):
    """
    Theorem 2: The query length that maximize the expected query segment accuracy score for a given event duration d_e and gamma.
    """
    return d_e*gamma*(2*gamma + np.sqrt(4*gamma**2 + 4*gamma + 2))/(2*gamma + 1)

def P_max(gamma):
    """
    Theorem 3: The maximum expected query segment accuracy score for a given gamma
    """
    return 2*gamma*(2*gamma - np.sqrt(4*gamma*(gamma + 1) + 2) + 1) + 1

def B_fix(T, d_e, gamma):
    """
    Theorem 4: The number of query segments that maximize the expected query segment accuracy score for a given event duration d_e, audio recording length T, and gamma.
    """
    return T / Q_max(d_e, gamma)

def TheoreticalLabelAccuracy(d_e, d_q, gamma, M, T):
    """
    Theorem 5: The label accuracy for a given event duration d_e, query duration d_q, presence criterion gamma, number events M, and audio recording length T.

    Args:
    - d_e: event duration
    - d_q: query duration
    - gamma: presence criterion
    - M: number of events
    - T: audio recording length

    Returns:
    - expected label accuracy
    """
    return -2*M*d_e**2*gamma**2/(T*d_q) + 2*M*d_e*gamma/T - M*d_q/T + 1

def TheoreticalLabelAccuracyMax(d_e, gamma, M, T):
    """
    The maximum label accuracy for a given event duration d_e, presence criterion gamma, number events M, and audio recording length T.

    Args:
    - d_e: event duration
    - gamma: presence criterion
    - M: number of events
    - T: audio recording length

    Returns:
    - maximum label accuracy
    """
    return -2*M*d_e*gamma*(2*gamma + 1)/(T*(2*gamma + np.sqrt(4*gamma**2 + 4*gamma + 2))) + 2*M*d_e*gamma/T - M*d_e*gamma*(2*gamma + np.sqrt(4*gamma**2 + 4*gamma + 2))/(T*(2*gamma + 1)) + 1