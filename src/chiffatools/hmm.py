"""
Code is courtesy of AustinRochford: https://github.com/AustinRochford/hmm/tree/master/python
"""
import numpy as np


class HMM:
    # constructor
    # transition_probs[i, j] is the probability of transitioning to state i from state j
    # emission_probs[i, j] is the probability of emitting emission j while in
    # state i

    def __init__(self, transition_probs, emission_probs):
        self._transition_probs = transition_probs
        self._emission_probs = emission_probs

    # accessors
    def emission_dist(self, emission):
        return self._emission_probs[:, emission]

    @property
    def num_states(self):
        return self._transition_probs.shape[0]

    @property
    def transition_probs(self):
        return self._transition_probs


# the Viterbi algorithm
def viterbi(hmm, initial_dist, emissions):
    probs = np.log(hmm.emission_dist(emissions[0]) * initial_dist)
    stack = []

    for emission in emissions[1:]:
        trans_probs = np.log(hmm.transition_probs) + np.row_stack(probs)
        max_col_ixs = np.argmax(trans_probs, axis=0)
        probs = np.log(hmm.emission_dist(emission)) + \
            trans_probs[max_col_ixs, np.arange(hmm.num_states)]

        stack.append(max_col_ixs)

    state_seq = [np.argmax(probs)]

    while stack:
        max_col_ixs = stack.pop()
        state_seq.append(max_col_ixs[state_seq[-1]])

    state_seq.reverse()

    return state_seq


# forward-backward algorithm
def backward(hmm, emissions):
    dist = uniform(hmm.num_states)
    dists = [dist]

    for emission in reversed(emissions):
        dist = backward_step(hmm, dist, emission)
        dists.append(dist)

    dists.reverse()

    return np.row_stack(dists)


def backward_step(hmm, dist, emission):
    return normalize(np.dot(hmm.transition_probs, np.dot(
        np.diagflat(hmm.emission_dist(emission)), dist.T)).T)


def forward_backward(hmm, initial_dist, emissions):
    forward_dists = forward(hmm, initial_dist, emissions)
    backward_dists = backward(hmm, emissions)

    return normalize(np.multiply(forward_dists, backward_dists))


def forward(hmm, initial_dist, emissions):
    dist = initial_dist
    dists = [dist]

    for emission in emissions:
        dist = forward_step(hmm, dist, emission)
        dists.append(dist)

    return np.row_stack(dists)


def forward_step(hmm, dist, emission):
    return normalize(
        np.dot(
            dist,
            np.dot(
                hmm.transition_probs,
                np.diagflat(
                    hmm.emission_dist(emission)))))


# related utilities
def modify_tuple(tuple_, ix, value):
    as_list = list(tuple_)
    as_list[ix] = value

    return tuple(as_list)


def normalize(array, axis=1):
    sum_shape = modify_tuple(array.shape, axis, 1)
    return array / np.reshape(np.sum(array, axis=axis), sum_shape)


def uniform(n):
    return normalize(np.ones((1, n)))


if __name__ == "__main__":
    # examples
    # from Wikipedia
    wiki_transition_probs = np.array(
        [[0.7, 0.4], [0.3, 0.6]])  # 0=Healthy, 1=Fever
    wiki_emissions = [2, 1, 0]
    wiki_emission_probs = np.array(
        [[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]])  # 0=Dizzy, 1=Cold, 2=Normal
    wiki_initial_dist = np.array([[0.6, 0.4]])
    wiki_hmm = HMM(wiki_transition_probs, wiki_emission_probs)
    print(viterbi(wiki_hmm, wiki_initial_dist, wiki_emissions))

    #print(forward(wiki_HMM, wiki_initial_dist, wiki_emissions))
    #print(backward(wiki_HMM, wiki_emissions))
    wiki_emission_probs = np.array([[0.9, 0.1], [0.2, 0.8]])
    wiki_initial_dist = np.array([[0.5, 0.5]])
    wiki_emissions = [0, 0, 1, 0, 0]
    wiki_transition_probs = np.array([[0.7, 0.3], [0.3, 0.7]])
    wiki_HMM = HMM(wiki_transition_probs, wiki_emission_probs)
    print(forward_backward(wiki_HMM, wiki_initial_dist, wiki_emissions))
