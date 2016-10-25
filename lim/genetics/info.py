# class NormalPhenotypeInfo(object):
#
#     def __init__(self, y):
#         self.y = y
#
#     def get_info(self):
#         lik = self.likelihood
#         t = [['Likelihood', lik[0].upper() + lik[1:]]]
#         if self.likelihood == 'normal':
#             t.append(['Phenotype mean', self.phenotype.mean()])
#             t.append(['Phenotype std', self.phenotype.std()])
#             mima = (self.phenotype.min(), self.phenotype.max())
#             t.append(['Phenotype (min, max)', mima])
#         return t
#
#
# class BinomialPhenotypeInfo(object):
#
#     def __init__(self, nsuccesses, ntrials):
#         self.nsuccesses = nsuccesses
#         self.ntrials = ntrials
#
#     def get_info(self):
#         lik = self.likelihood
#         t = [['Likelihood', lik[0].upper() + lik[1:]]]
#         if self.likelihood == 'normal':
#             t.append(['Phenotype mean', self.phenotype.mean()])
#             t.append(['Phenotype std', self.phenotype.std()])
#             mima = (self.phenotype.min(), self.phenotype.max())
#             t.append(['Phenotype (min, max)', mima])
#         return t
