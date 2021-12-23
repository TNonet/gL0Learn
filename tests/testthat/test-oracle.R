library("gL0Learn")
library('pracma')


test_that("gL0Learn.oracle_prox properly fails properly", {
    N = 3
    M = 4
    
    # No Bounds
    
    expect_equal(gL0Learn.oracle_prox(theta=5, l0=1, l2=1), 5/3)
    expect_equal(gL0Learn.oracle_prox(1:N, l0=1, l2=1), c(0, 0, 1))
    expect_equal(gL0Learn.oracle_prox(N:(2*N - 1), l0=1:N, l2=1:N), c(1, 0, 0))
    expect_error(gL0Learn.oracle_prox(1:M, l0=1:N, l2=1:N))
    expect_error(gL0Learn.oracle_prox(1:N, l0=1:M, l2=1:N))
    expect_error(gL0Learn.oracle_prox(1:N, l0=1:N, l2=1:M))
    expect_equal(gL0Learn.oracle_prox(4, l0=1, l1=1, l2=1), 1)
    expect_equal(gL0Learn.oracle_prox(N:(2*N-1), l0=1, l1=1, l2=1), c(0, 1, 4/3))
    expect_error(gL0Learn.oracle_prox(1, l0=1:N, l2=1:N))
    expect_error(gL0Learn.oracle_prox(1, l0=1, l2=1:N))
    expect_error(gL0Learn.oracle_prox(1, l0=1:N, l2=1))
    expect_error(gL0Learn.oracle_prox(1, l0=1:N, l2=1:M))
    expect_error(gL0Learn.oracle_prox(1:N, l0=1:N, l1 = 1, l2=1:M))
    expect_error(gL0Learn.oracle_prox(1:N, l0=1:N, l1 = 1, l2=1:N))
    expect_error(gL0Learn.oracle_prox(1, l0=1, l1 = 1:N, l2=1))
    expect_error(gL0Learn.oracle_prox(1:N, l0=1, l1 = 1:N, l2=1))
    expect_error(gL0Learn.oracle_prox(1, l0=1:N, l1 = 1:N, l2=1:N))
    expect_equal(gL0Learn.oracle_prox((2*N):(3*N-1), l0=1:N, l1=1:N, l2=1:N), c(5/3, 1, 0))
    expect_equal(gL0Learn.oracle_prox(matrix(1:4, 2, 2), 1, 1, 1), matrix(c(0, 0, 0, 1), 2, 2))
    expect_error(gL0Learn.oracle_prox(matrix(1:4, 2, 2), 1:N, 1, 1))
    expect_error(gL0Learn.oracle_prox(matrix(1:4, 2, 2), 1:N, 1:N, 1:N))
    expect_equal(gL0Learn.oracle_prox(matrix(4:7, 2, 2), matrix(1:4, 2, 2), matrix(1:4, 2, 2), matrix(1:4, 2, 2)), matrix(c(1, 0, 0, 0), 2, 2))
    expect_error(gL0Learn.oracle_prox(matrix(1:4, 2, 2), matrix(1:9, 3, 3), matrix(1:9, 3, 3), matrix(1:9, 3, 3)))
    
    # Scalar Bounds
    
    expect_equal(gL0Learn.oracle_prox(theta=5, l0=1, l2=1, lows=-1, highs=1.2), 1.2)
    expect_equal(gL0Learn.oracle_prox(2:4, l0=1, l2=1, lows=-1, highs=1.2), c(0, 1.0, 1.2))
    expect_equal(gL0Learn.oracle_prox(5:7, l0=1:N, l2=1:N, lows=-1, highs=1.2), c(1.2, 1.2, 1.0))
    expect_error(gL0Learn.oracle_prox(1:M, l0=1:N, l2=1:N, lows=-1, highs=10))
    expect_error(gL0Learn.oracle_prox(1:N, l0=1:M, l2=1:N, lows=-1, highs=10))
    expect_error(gL0Learn.oracle_prox(1:N, l0=1:N, l2=1:M, lows=-1, highs=10))
    expect_error(gL0Learn.oracle_prox(1, l0=1:N, l2=1:N, lows=-1, highs=10))
    expect_error(gL0Learn.oracle_prox(1, l0=1, l2=1:N, lows=-1, highs=10))
    expect_error(gL0Learn.oracle_prox(1, l0=1:N, l2=1, lows=-1, highs=10))
    expect_error(gL0Learn.oracle_prox(1, l0=1:N, l2=1:M, lows=-1, highs=10))
    expect_error(gL0Learn.oracle_prox(1:N, l0=1:N, l1 = 1, l2=1:M, lows=-1, highs=10))
    expect_error(gL0Learn.oracle_prox(1:N, l0=1:N, l1 = 1, l2=1:N, lows=-1, highs=10))
    expect_error(gL0Learn.oracle_prox(1, l0=1, l1 = 1:N, l2=1, lows=-1, highs=10))
    expect_error(gL0Learn.oracle_prox(1:N, l0=1, l1 = 1:N, l2=1, lows=-1, highs=10))
    expect_error(gL0Learn.oracle_prox(1, l0=1:N, l1 = 1:N, l2=1:N, lows=-1, highs=10))
    expect_equal(gL0Learn.oracle_prox(theta=matrix(5:8, 2, 2), l0=1, l2=1, lows=-1, highs=2), matrix(c(5/3, 2, 2, 2), 2, 2))
    expect_equal(gL0Learn.oracle_prox(theta=matrix(5:8, 2, 2), l0=matrix(1:4, 2, 2), l2=matrix(1:4, 2, 2), lows=-1, highs=2), matrix(c(5/3, 1.2, 1, 0), 2, 2))
    
    
    # Vector Bounds
    expect_error(gL0Learn.oracle_prox(theta=1:N, l0=1, l2=1, lows=-1, highs=1:N))
    expect_error(gL0Learn.oracle_prox(theta=1:N, l0=1, l2=1, highs=1:N))
    expect_error(gL0Learn.oracle_prox(theta=1:N, l0=1, l2=1, lows=-1:-N,  highs=1))
    expect_error(gL0Learn.oracle_prox(theta=1:N, l0=1, l2=1, lows=-1:-M, highs=1:N))
    
    
    expect_error(gL0Learn.oracle_prox(theta=1, l0=1, l2=1, lows=-1:-N, highs=1:N))
    expect_equal(gL0Learn.oracle_prox(1:N, l0=1, l2=1, lows=-1:-N, highs=rep(0.8, 3)), c(0, 0, 0.8))
    expect_equal(gL0Learn.oracle_prox(5:7, l0=1:N, l2=1:N, lows=-1:-N, highs=1:N), c(1.0, 1.2, 1.0))
    expect_error(gL0Learn.oracle_prox(1:M, l0=1:N, l2=1:N, lows=-1:-N, highs=1:N))
    expect_error(gL0Learn.oracle_prox(1:N, l0=1:M, l2=1:N, lows=-1:-N, highs=1:N))
    expect_error(gL0Learn.oracle_prox(1:N, l0=1:N, l2=1:M, lows=-1:-N, highs=1:N))
    expect_error(gL0Learn.oracle_prox(1, l0=1, l1=1, l2=1, lows=-1:-N, highs=1:N))
    expect_error(gL0Learn.oracle_prox(1:N, l0=1:N, l2=1:N, lows=-1:-N, highs=1:M))
    expect_error(gL0Learn.oracle_prox(1:N, l0=1:N, l1 = 1, l2=1:M, lows=-1:-N, highs=1:N))
    expect_error(gL0Learn.oracle_prox(1:N, l0=1:N, l1 = 1, l2=1:N, lows=-1:-N, highs=1:N))
    expect_error(gL0Learn.oracle_prox(1, l0=1, l1 = 1:N, l2=1, lows=-1:-N, highs=1:N))
    expect_error(gL0Learn.oracle_prox(1:N, l0=1, l1 = 1:N, l2=1, lows=-1:-N, highs=1:N))
    expect_error(gL0Learn.oracle_prox(1, l0=1:N, l1 = 1:N, l2=1:N, lows=-1:-N, highs=1:N))
    expect_error(gL0Learn.oracle_prox(matrix(1, 2, 2), l0=matrix(1, 2, 2), l1 = matrix(1, 2, 2), l2=matrix(1, 2, 2), lows=-1:-N, highs=1:N))
    expect_error(gL0Learn.oracle_prox(matrix(1, 2, 2), l0=matrix(1, 2, 2), l1 = matrix(1, 2, 2), l2=matrix(1, 2, 2), lows=-1:-N, highs=1:N))
    
    ## Matrix bounds
    expect_error(gL0Learn.oracle_prox(1, l0=1, l1 = 1, l2=1, lows=-matrix(1, 2, 2), highs=matrix(1, 2, 2)))
    expect_error(gL0Learn.oracle_prox(1, l0=1, l1 = 1, l2=1, lows=-matrix(1, 2, 2), highs=1))
    expect_error(gL0Learn.oracle_prox(matrix(1, 2, 2), l0=1, l1 = 1, l2=1, lows=-matrix(1, 2, 2), highs=1))
    expect_error(gL0Learn.oracle_prox(matrix(1, 2, 2), l0=1, l1 = 1, l2=1, lows=-matrix(1, 3, 3), highs=matrix(1, 3, 3)))
    expect_equal(gL0Learn.oracle_prox(matrix(1:4, 2, 2), l0=1, l1 = 1, l2=1, lows=matrix(0, 2, 2), highs=matrix(0.8, 2, 2)),
                 matrix(c(0, 0, 0, 0.8), 2, 2))
})

NUM_RUNS = 50
RUN_SIZE = 10
M = 2

test_that("prox finds similar solutions to linear search for L0L1L2 Bounds", {
    N = RUN_SIZE
    
    for (i in 1:NUM_RUNS){
        theta_opt <- matrix(rnorm(N*M), N, M) 
        lows <- - matrix(runif(N*M), N, M) 
        highs <-  matrix(runif(N*M), N, M) 
        l0 <- matrix(runif(N*M), N, M) 
        l1 <- matrix(runif(N*M), N, M) 
        l2 <- matrix(runif(N*M), N, M) 
        
        oracle_matrix <- gL0Learn.oracle_prox(theta_opt, l0, l1, l2, lows, highs)
        
        for (j in 1:M){
            # Test L0L1L2 with Bounds
            oracle_vector <- gL0Learn.oracle_prox(theta_opt[j,],
                                                  l0[j,],
                                                  l1[j,],
                                                  l2[j,],
                                                  lows[j,],
                                                  highs[j,])
            
            test_func <- function(k) {gL0Learn.oracle_prox(theta_opt[j,k],
                                                           l0[j,k],
                                                           l1[j,k],
                                                           l2[j,k],
                                                           lows[j,k],
                                                           highs[j,k])}
            
            
            oracle_vector2 <- sapply(1:M, test_func)
            
            test_func_2 <- function(k) {gL0Learn.linear_search(theta_opt[j,k],
                                                               l0[j,k],
                                                               l1[j,k],
                                                               l2[j,k],
                                                               lows[j,k],
                                                               highs[j,k])}
            
            search_vector <- sapply(1:M, test_func_2)
            
            for (k in 1:M){
                info = sprintf("error found with theta=%f, l0=%f, l1=%f, l2=%f, lows=%f, highs=%f", 
                               theta_opt[j,k],
                               l0[j,k],
                               l1[j,k],
                               l2[j,k],
                               lows[j,k],
                               highs[j,k])
                expect_equal(oracle_vector[k], search_vector[k], tolerance = 1e-6, info=info)
                expect_equal(oracle_vector2[k], search_vector[k], tolerance = 1e-6, info=info)
                expect_equal(oracle_matrix[j, k], search_vector[k], tolerance = 1e-6, info=info)
            }
        }
    }
})


test_that("prox finds similar solutions to linear search for L0L1L2 NoBounds", {
    N = RUN_SIZE
    M = 2
    
    for (i in 1:NUM_RUNS){
        theta_opt <- matrix(rnorm(N*M), N, M) 
        lows <- - Inf*matrix(1, N, M) 
        highs <-  Inf*matrix(1, N, M) 
        l0 <- matrix(runif(N*M), N, M) 
        l1 <- matrix(runif(N*M), N, M) 
        l2 <- matrix(runif(N*M), N, M) 
        
        oracle_matrix <- gL0Learn.oracle_prox(theta_opt, l0, l1, l2, lows, highs)
        
        for (j in 1:M){
            # Test L0L1L2 with Bounds
            oracle_vector <- gL0Learn.oracle_prox(theta_opt[j,],
                                                  l0[j,],
                                                  l1[j,],
                                                  l2[j,],
                                                  lows[j,],
                                                  highs[j,])
            
            test_func <- function(k) {gL0Learn.oracle_prox(theta_opt[j,k],
                                                           l0[j,k],
                                                           l1[j,k],
                                                           l2[j,k],
                                                           lows[j,k],
                                                           highs[j,k])}
            
            oracle_vector2 <- sapply(1:M, test_func)
            
            search_func <- function(k) {gL0Learn.linear_search(theta_opt[j,k],
                                                               l0[j,k],
                                                               l1[j,k],
                                                               l2[j,k],
                                                               min(0, theta_opt[j, k]),
                                                               max(0, theta_opt[j, k]))}
            
            search_vector <- sapply(1:M, search_func)
            
            for (k in 1:M){
                info = sprintf("error found with theta=%f, l0=%f, l1=%f, l2=%f, lows=%f, highs=%f", 
                               theta_opt[j,k],
                               l0[j,k],
                               l1[j,k],
                               l2[j,k],
                               lows[j,k],
                               highs[j,k])
                expect_equal(oracle_vector[k], search_vector[k], tolerance = 1e-6, info=info)
                expect_equal(oracle_vector2[k], search_vector[k], tolerance = 1e-6, info=info)
                expect_equal(oracle_matrix[j, k], search_vector[k], tolerance = 1e-6, info=info)
            }
        }
    }
})

test_that("prox finds similar solutions to linear search for L0L2 NoBounds", {
    N = RUN_SIZE
    
    for (i in 1:NUM_RUNS){
        theta_opt <- matrix(rnorm(N*M), N, M)
        lows <- - Inf*matrix(1, N, M) 
        highs <-  Inf*matrix(1, N, M) 
        l0 <- matrix(runif(N*M), N, M) 
        l1 <- matrix(0, N, M) 
        l2 <- matrix(runif(N*M), N, M) 
        
        oracle_matrix <- gL0Learn.oracle_prox(theta_opt, l0, l1, l2, -Inf, +Inf)
        oracle_matrix_0 <- gL0Learn.oracle_prox(theta_opt, l0, 0, l2, lows, highs)
        
        for (j in 1:M){
            # Test L0L1L2 with Bounds
            oracle_vector <- gL0Learn.oracle_prox(theta_opt[j,],
                                                  l0[j,],
                                                  l1[j,],
                                                  l2[j,],
                                                  lows[j,],
                                                  highs[j,])
            oracle_vector_0 <- gL0Learn.oracle_prox(theta_opt[j,],
                                                  l0[j,],
                                                  0,
                                                  l2[j,],
                                                  lows[j,],
                                                  highs[j,])
            
            test_func <- function(k) {gL0Learn.oracle_prox(theta_opt[j,k],
                                                           l0[j,k],
                                                           l1[j,k],
                                                           l2[j,k],
                                                           lows[j,k],
                                                           highs[j,k])}
            
            oracle_vector2 <- sapply(1:M, test_func)
            
            test_func_0 <- function(k) {gL0Learn.oracle_prox(theta_opt[j,k],
                                                           l0[j,k],
                                                           0,
                                                           l2[j,k],
                                                           lows[j,k],
                                                           highs[j,k])}
            
            oracle_vector_0_2 <- sapply(1:M, test_func_0)
            
            search_func <- function(k) {gL0Learn.linear_search(theta_opt[j,k],
                                                               l0[j,k],
                                                               l1[j,k],
                                                               l2[j,k],
                                                               min(0, theta_opt[j, k]),
                                                               max(0, theta_opt[j, k]))}
            
            search_vector <- sapply(1:M, search_func)
            
            for (k in 1:M){
                info = sprintf("error found with theta=%f, l0=%f, l1=%f, l2=%f, lows=%f, highs=%f", 
                               theta_opt[j,k],
                               l0[j,k],
                               l1[j,k],
                               l2[j,k],
                               lows[j,k],
                               highs[j,k])
                expect_equal(oracle_vector[k], search_vector[k], tolerance = 1e-6, info=info)
                expect_equal(oracle_vector_0[k], search_vector[k], tolerance = 1e-6, info=info)
                expect_equal(oracle_vector2[k], search_vector[k], tolerance = 1e-6, info=info)
                expect_equal(oracle_vector_0_2[k], search_vector[k], tolerance = 1e-6, info=info)
                expect_equal(oracle_matrix[j, k], search_vector[k], tolerance = 1e-6, info=info)
                expect_equal(oracle_matrix_0[j, k], search_vector[k], tolerance = 1e-6, info=info)
            }
        }
    }
})


test_that("prox finds similar solutions to linear search for L0L2 Bounds", {
    N = RUN_SIZE
    
    for (i in 1:NUM_RUNS){
        theta_opt <- matrix(rnorm(N*M), N, M) 
        lows <- -matrix(runif(N*M), N, M) 
        highs <- matrix(runif(N*M), N, M) 
        l0 <- matrix(runif(N*M), N, M) 
        l1 <- matrix(0, N, M) 
        l2 <- matrix(runif(N*M), N, M) 
        
        oracle_matrix <- gL0Learn.oracle_prox(theta_opt, l0, l1, l2, lows, highs)
        oracle_matrix_0 <- gL0Learn.oracle_prox(theta_opt, l0, 0, l2, lows, highs)
        
        for (j in 1:M){
            # Test L0L1L2 with Bounds
            oracle_vector <- gL0Learn.oracle_prox(theta_opt[j,],
                                                  l0[j,],
                                                  l1[j,],
                                                  l2[j,],
                                                  lows[j,],
                                                  highs[j,])
            oracle_vector_0 <- gL0Learn.oracle_prox(theta_opt[j,],
                                                    l0[j,],
                                                    0,
                                                    l2[j,],
                                                    lows[j,],
                                                    highs[j,])
            
            test_func <- function(k) {gL0Learn.oracle_prox(theta_opt[j,k],
                                                           l0[j,k],
                                                           l1[j,k],
                                                           l2[j,k],
                                                           lows[j,k],
                                                           highs[j,k])}
            
            oracle_vector2 <- sapply(1:M, test_func)
            
            test_func_0 <- function(k) {gL0Learn.oracle_prox(theta_opt[j,k],
                                                             l0[j,k],
                                                             0,
                                                             l2[j,k],
                                                             lows[j,k],
                                                             highs[j,k])}
            
            oracle_vector_0_2 <- sapply(1:M, test_func_0)
            
            search_func <- function(k) {gL0Learn.linear_search(theta_opt[j,k],
                                                               l0[j,k],
                                                               l1[j,k],
                                                               l2[j,k],
                                                               lows[j,k],
                                                               highs[j,k])}
            
            search_vector <- sapply(1:M, search_func)
            
            for (k in 1:M){
                info = sprintf("error found with theta=%f, l0=%f, l1=%f, l2=%f, lows=%f, highs=%f", 
                               theta_opt[j,k],
                               l0[j,k],
                               l1[j,k],
                               l2[j,k],
                               lows[j,k],
                               highs[j,k])
                expect_equal(oracle_vector[k], search_vector[k], tolerance = 1e-6, info=info)
                expect_equal(oracle_vector_0[k], search_vector[k], tolerance = 1e-6, info=info)
                expect_equal(oracle_vector2[k], search_vector[k], tolerance = 1e-6, info=info)
                expect_equal(oracle_vector_0_2[k], search_vector[k], tolerance = 1e-6, info=info)
                expect_equal(oracle_matrix[j, k], search_vector[k], tolerance = 1e-6, info=info)
                expect_equal(oracle_matrix_0[j, k], search_vector[k], tolerance = 1e-6, info=info)
            }
        }
    }
})
