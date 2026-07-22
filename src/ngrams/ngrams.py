
import string

import numpy as np

###
# FUNCTIONS
###

def make_eta_trigram(mu_bigram_left, mu_bigram_right):
    return np.array([mu_bigram_left[0], mu_bigram_left[1], 0.5*mu_bigram_left[2]+0.5*mu_bigram_right[0], mu_bigram_right[1], mu_bigram_right[2]])

def make_eta_bigram(mu_unigram_left, mu_transit, mu_unigram_right):
    return np.array([mu_unigram_left[0], mu_transit, mu_unigram_right[0]])


###
# Generate Data
### 

# seed
np.random.seed(0)

# general things
alphabet = list(string.ascii_uppercase) # A to Z

mu_Omega = np.array([0.3]) # TODO: This should be called mu_unigram
V_Omega = np.array([[0.1**2]]) # TODO: This should be called V_unigram
V_bigram = 0.1*np.eye(3)
V_trigram = 0.1*np.eye(5)

Sigma_trigram = 0.1*np.eye(5)

mu_transit = 0.0 

# data generation
M=1000

# THE MODEL

mus_unigram = {}
for character in alphabet:
    mus_unigram[character] = np.random.multivariate_normal(mean=mu_Omega, cov=V_Omega)

etas_bigram = {}
mus_bigram = {}
for left_character in alphabet:
    for right_character in alphabet:
        bigram = left_character + right_character
        etas_bigram[bigram] =  make_eta_bigram(mus_unigram[left_character], mu_transit, mus_unigram[right_character])
        mus_bigram[bigram] = np.random.multivariate_normal(mean=etas_bigram[bigram], cov=V_bigram)

etas_trigram = {}
mus_trigram = {}
for left_character in alphabet:
    for mid_character in alphabet:
        for right_character in alphabet:
            trigram = left_character + mid_character + right_character
            bigram_left = left_character + mid_character
            bigram_right = mid_character + right_character
            etas_trigram[trigram] =  make_eta_trigram(mus_bigram[bigram_left], mus_bigram[bigram_right])
            mus_trigram[trigram] = np.random.multivariate_normal(mean=etas_trigram[trigram], cov=V_trigram)


y_THE = np.random.multivariate_normal(mean=mus_trigram["THE"], cov=Sigma_trigram, size=M)


# TODO: Use Psi matrices explicitly in the make_eta functions above.
Phi_left_2 = np.array([[1],[0.5*mu_transit],[0]])
Phi_right_2 = np.array([[0],[0.5*mu_transit],[1]])

Phi_left_3 = np.array([[1,0,0],[0,1,0],[0,0,0.5],[0,0,0],[0,0,0]])
Phi_right_3 = np.array([[0,0,0],[0,0,0],[0.5,0,0],[0,1,0],[0,0,1]])

### 
# Inference -- Gibbs sampler
###

### Initialization

mu_hat_T = 0.0
mu_hat_H = 0.0
mu_hat_E = 0.0
mu_hat_TH = 0.0
mu_hat_HE = 0.0
mu_hat_THE = 0.0

### Precompute things

y_THE_bar = np.mean(y_THE,0)

V_unigram_inv = np.linalg.inv(V_Omega)
V_bigram_inv = np.linalg.inv(V_bigram)
V_trigram_inv = np.linalg.inv(V_trigram)
Sigma_trigram_inv = np.linalg.inv(Sigma_trigram)

### Give true conditionals, since we're currently just checking inference functions
mu_THE = mus_trigram["THE"]
mu_TH = mus_bigram["TH"]
mu_HE= mus_bigram["HE"]
mu_T = mus_unigram["T"]
mu_H = mus_unigram["H"]
mu_E = mus_unigram["E"]

### Check Inference for mu_hat_THE (by giving it the true value of parents)

eta_THE = make_eta_trigram(mu_TH, mu_HE)
A_THE = V_trigram_inv + M* Sigma_trigram_inv # precision 
b_THE =  V_trigram_inv @ eta_THE + M*  Sigma_trigram_inv @ y_THE_bar # precision-weighted mean

# TODO: Compute the below upfroint, only once.
V_THE_post = np.linalg.inv(A_THE)
eta_THE_post = V_THE_post @ b_THE

# take lots of samples for now
S= 1000 
mu_THE_samples = np.random.multivariate_normal(mean=eta_THE_post, cov=V_THE_post, size=S)
mean_of_mu_THE_samples = np.mean(mu_THE_samples,0) # this is close to the true value! Hurrah! 
print(f"True value of mu_THE: {mu_THE}")
print(f"Posterior mean of mu_THE: {mean_of_mu_THE_samples}")

# TODO: Write the above manual check as an assert.

### Check Inference for mu_hat_TH (by giving it the true value of parents)

eta_TH =  make_eta_bigram(mu_T, mu_transit, mu_H)


# A_TH = V_bigram_inv + Phi_left_3.T @ V_trigram_inv @  Phi_left_3 # precision 
# b_TH =  V_bigram_inv @ eta_TH + Phi_left_3.T @ V_trigram_inv @ (mu_THE -  Phi_right_3 @ mu_HE) # precision-weighted mean

A_TH = V_bigram_inv + Phi_left_3.T @ V_trigram_inv @  Phi_left_3 # precision 
b_TH_ =  V_bigram_inv @ eta_TH 
for character in alphabet:
    expansion = "TH"+character  #e.g. THE
    complement = "H"+character  #e.g HE
    mu_expansion = mus_trigram[expansion]
    mu_complement = mus_bigram[complement]
    b_TH_ += Phi_left_3.T @ V_trigram_inv @ (mu_expansion -  Phi_right_3 @ mu_complement)
b_TH = b_TH_ 


# TODO: Compute the below upfroint, only once.
V_TH_post = np.linalg.inv(A_TH)
eta_TH_post = V_TH_post @ b_TH

# take lots of samples for now
S= 1000 
mu_TH_samples = np.random.multivariate_normal(mean=eta_TH_post, cov=V_TH_post, size=S)
mean_of_mu_TH_samples = np.mean(mu_TH_samples,0)  
print(f"True value of mu_TH: {mu_TH}")
print(f"Posterior mean of mu_TH: {mean_of_mu_TH_samples}")

# TODO: Write the above manual check as an assert.



# ### Check Inference for mu_hat_T (by giving it the true value of parents)
# TODO: Update this section. 

# eta_T = mu_Omega

# A_T = V_unigram_inv +  Phi_left_2.T@V_bigram_inv@Phi_left_2 + Phi_right_2.T@V_bigram_inv@Phi_right_2  # precision 
# b_T =  V_unigram_inv @ eta_T + Phi_right_2.T @ V_bigram_inv @ (mu_TH -  Phi_left_2 @ mu_H) # precision-weighted mean

# # TODO: Compute the below upfront, only once.
# V_T_post = np.linalg.inv(A_T)
# eta_T_post = V_T_post @ b_T

# # take lots of samples for now
# S= 1000
# mu_T_samples = np.random.multivariate_normal(mean=eta_T_post, cov=V_T_post, size=S)
# mean_of_mu_T_samples = np.mean(mu_T_samples,0)  
# print(f"True value of mu_T: {mu_T}")
# print(f"Posterior mean of mu_T: {mean_of_mu_T_samples}")

# # TODO: Write the above manual check as an assert.

