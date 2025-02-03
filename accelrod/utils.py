def get_power_of_two_sequence(N):
    """
    Returns a list of integers where each number is a power of 2, the largest number <= N.

    Args:
        N (int): The upper limit of the sequence

    Returns:
        list: A list of powers of 2 up to N
    """
    sequence = []
    power = 0
    while 2**power <= N:
        sequence.append(2**power)
        power += 1
    return sequence
