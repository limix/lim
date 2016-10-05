
var = lim.genetics.variance.normal_decomposition(y, X, [G0, G1, G2])
var = lim.genetics.variance.normal_decomposition(y, X, dict(Genetic=G0,
                                                            Cage=G1,
                                                            Weather=G2))
var = lim.genetics.variance.normal_decomposition(y, X, dict(Genetic=(G0, False),
                                                            Cage=(K1, True),
                                                            Weather=(K2, True))
