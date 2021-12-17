library("templateFunctions")
library('pracma')


test_that("resolve_func_name properly resolves test functions", {
    N = 10
    M = 8
    
    # No Bounds
    
    expect_equal(gL0Learn.resolve_func_name(theta=1, l0=1, l2=1), "test_OracleScalarL0L2NoBounds_prox_double")
    expect_equal(gL0Learn.resolve_func_name(1:N, l0=1, l2=1), "test_OracleScalarL0L2NoBounds_prox_vec")
    expect_equal(gL0Learn.resolve_func_name(1:N, l0=1:N, l2=1:N), "test_OracleVectorL0L2NoBounds_prox_vec")
    expect_error(gL0Learn.resolve_func_name(1:M, l0=1:N, l2=1:N))
    expect_error(gL0Learn.resolve_func_name(1:N, l0=1:M, l2=1:N))
    expect_error(gL0Learn.resolve_func_name(1:N, l0=1:N, l2=1:M))
    expect_equal(gL0Learn.resolve_func_name(1, l0=1, l1=1, l2=1), "test_OracleScalarL0L1L2NoBounds_prox_double")
    expect_equal(gL0Learn.resolve_func_name(1:N, l0=1, l1=1, l2=1), "test_OracleScalarL0L1L2NoBounds_prox_vec")
    expect_equal(gL0Learn.resolve_func_name(1:N, l0=1, l1=1, l2=1), "test_OracleScalarL0L1L2NoBounds_prox_vec")
    expect_error(gL0Learn.resolve_func_name(1, l0=1:N, l2=1:N))
    expect_error(gL0Learn.resolve_func_name(1, l0=1, l2=1:N))
    expect_error(gL0Learn.resolve_func_name(1, l0=1:N, l2=1))
    expect_error(gL0Learn.resolve_func_name(1, l0=1:N, l2=1:M))
    expect_error(gL0Learn.resolve_func_name(1:N, l0=1:N, l1 = 1, l2=1:M))
    expect_error(gL0Learn.resolve_func_name(1:N, l0=1:N, l1 = 1, l2=1:N))
    expect_error(gL0Learn.resolve_func_name(1, l0=1, l1 = 1:N, l2=1))
    expect_error(gL0Learn.resolve_func_name(1:N, l0=1, l1 = 1:N, l2=1))
    expect_error(gL0Learn.resolve_func_name(1, l0=1:N, l1 = 1:N, l2=1:N))
    expect_equal(gL0Learn.resolve_func_name(1:N, l0=1:N, l1=1:N, l2=1:N), "test_OracleVectorL0L1L2NoBounds_prox_vec")
    
    # Scalar Bounds
    
    expect_equal(gL0Learn.resolve_func_name(theta=1, l0=1, l2=1, lows=-1, highs=10), "test_OracleScalarL0L2ScalarBounds_prox_double")
    expect_equal(gL0Learn.resolve_func_name(1:N, l0=1, l2=1, lows=-1, highs=10), "test_OracleScalarL0L2ScalarBounds_prox_vec")
    expect_equal(gL0Learn.resolve_func_name(1:N, l0=1:N, l2=1:N, lows=-1, highs=10), "test_OracleVectorL0L2ScalarBounds_prox_vec")
    expect_error(gL0Learn.resolve_func_name(1:M, l0=1:N, l2=1:N, lows=-1, highs=10))
    expect_error(gL0Learn.resolve_func_name(1:N, l0=1:M, l2=1:N, lows=-1, highs=10))
    expect_error(gL0Learn.resolve_func_name(1:N, l0=1:N, l2=1:M, lows=-1, highs=10))
    expect_equal(gL0Learn.resolve_func_name(1, l0=1, l1=1, l2=1, lows=-1, highs=10), "test_OracleScalarL0L1L2ScalarBounds_prox_double")
    expect_equal(gL0Learn.resolve_func_name(1:N, l0=1, l1=1, l2=1, lows=-1, highs=10), "test_OracleScalarL0L1L2ScalarBounds_prox_vec")
    expect_equal(gL0Learn.resolve_func_name(1:N, l0=1, l1=1, l2=1, lows=-1, highs=10), "test_OracleScalarL0L1L2ScalarBounds_prox_vec")
    expect_error(gL0Learn.resolve_func_name(1, l0=1:N, l2=1:N, lows=-1, highs=10))
    expect_error(gL0Learn.resolve_func_name(1, l0=1, l2=1:N, lows=-1, highs=10))
    expect_error(gL0Learn.resolve_func_name(1, l0=1:N, l2=1, lows=-1, highs=10))
    expect_error(gL0Learn.resolve_func_name(1, l0=1:N, l2=1:M, lows=-1, highs=10))
    expect_error(gL0Learn.resolve_func_name(1:N, l0=1:N, l1 = 1, l2=1:M, lows=-1, highs=10))
    expect_error(gL0Learn.resolve_func_name(1:N, l0=1:N, l1 = 1, l2=1:N, lows=-1, highs=10))
    expect_error(gL0Learn.resolve_func_name(1, l0=1, l1 = 1:N, l2=1, lows=-1, highs=10))
    expect_error(gL0Learn.resolve_func_name(1:N, l0=1, l1 = 1:N, l2=1, lows=-1, highs=10))
    expect_error(gL0Learn.resolve_func_name(1, l0=1:N, l1 = 1:N, l2=1:N, lows=-1, highs=10))
    expect_equal(gL0Learn.resolve_func_name(1:N, l0=1:N, l1=1:N, l2=1:N, lows=-1, highs=10), "test_OracleVectorL0L1L2ScalarBounds_prox_vec")
    
    # Vector Bounds
    expect_error(gL0Learn.resolve_func_name(theta=1:N, l0=1, l2=1, lows=-1, highs=1:N))
    expect_error(gL0Learn.resolve_func_name(theta=1:N, l0=1, l2=1, highs=1:N))
    expect_error(gL0Learn.resolve_func_name(theta=1:N, l0=1, l2=1, lows=-1:-M, highs=1:N))
    
    
    expect_error(gL0Learn.resolve_func_name(theta=1, l0=1, l2=1, lows=-1:-N, highs=1:N))
    expect_equal(gL0Learn.resolve_func_name(1:N, l0=1, l2=1, lows=-1:-N, highs=1:N), "test_OracleScalarL0L2VectorBounds_prox_vec")
    expect_equal(gL0Learn.resolve_func_name(1:N, l0=1:N, l2=1:N, lows=-1:-N, highs=1:N), "test_OracleVectorL0L2VectorBounds_prox_vec")
    expect_error(gL0Learn.resolve_func_name(1:M, l0=1:N, l2=1:N, lows=-1:-N, highs=1:N))
    expect_error(gL0Learn.resolve_func_name(1:N, l0=1:M, l2=1:N, lows=-1:-N, highs=1:N))
    expect_error(gL0Learn.resolve_func_name(1:N, l0=1:N, l2=1:M, lows=-1:-N, highs=1:N))
    expect_error(gL0Learn.resolve_func_name(1, l0=1, l1=1, l2=1, lows=-1:-N, highs=1:N))
    expect_equal(gL0Learn.resolve_func_name(1:N, l0=1, l1=1, l2=1, lows=-1:-N, highs=1:N), "test_OracleScalarL0L1L2VectorBounds_prox_vec")
    expect_equal(gL0Learn.resolve_func_name(1:N, l0=1, l1=1, l2=1, lows=-1:-N, highs=1:N), "test_OracleScalarL0L1L2VectorBounds_prox_vec")
    expect_error(gL0Learn.resolve_func_name(1, l0=1:N, l2=1:N, lows=-1:-N, highs=1:N))
    expect_error(gL0Learn.resolve_func_name(1, l0=1, l2=1:N, lows=-1:-N, highs=1:N))
    expect_error(gL0Learn.resolve_func_name(1, l0=1:N, l2=1, lows=-1:-N, highs=1:N))
    expect_error(gL0Learn.resolve_func_name(1, l0=1:N, l2=1:M, lows=-1:-N, highs=1:N))
    expect_error(gL0Learn.resolve_func_name(1:N, l0=1:N, l1 = 1, l2=1:M, lows=-1:-N, highs=1:N))
    expect_error(gL0Learn.resolve_func_name(1:N, l0=1:N, l1 = 1, l2=1:N, lows=-1:-N, highs=1:N))
    expect_error(gL0Learn.resolve_func_name(1, l0=1, l1 = 1:N, l2=1, lows=-1:-N, highs=1:N))
    expect_error(gL0Learn.resolve_func_name(1:N, l0=1, l1 = 1:N, l2=1, lows=-1:-N, highs=1:N))
    expect_error(gL0Learn.resolve_func_name(1, l0=1:N, l1 = 1:N, l2=1:N, lows=-1:-N, highs=1:N))
    expect_equal(gL0Learn.resolve_func_name(1:N, l0=1:N, l1=1:N, l2=1:N, lows=-1:-N, highs=1:N), "test_OracleVectorL0L1L2VectorBounds_prox_vec")
    
})

# typedef Oracle<ScalarPenaltyL0L2, NoBounds>       OracleScalarL0L2NoBounds;
# typedef Oracle<VectorPenaltyL0L2, NoBounds>       OracleVectorL0L2NoBounds;
# typedef Oracle<ScalarPenaltyL0L1L2, NoBounds>     OracleScalarL0L1L2NoBounds;
# typedef Oracle<VectorPenaltyL0L1L2, NoBounds>     OracleVectorL0L1L2NoBounds;
# typedef Oracle<ScalarPenaltyL0L2, ScalarBounds>   OracleScalarL0L2ScalarBounds;
# typedef Oracle<VectorPenaltyL0L2, ScalarBounds>   OracleVectorL0L2ScalarBounds;
# typedef Oracle<ScalarPenaltyL0L1L2, ScalarBounds> OracleScalarL0L1L2ScalarBounds;
# typedef Oracle<VectorPenaltyL0L1L2, ScalarBounds> OracleVectorL0L1L2ScalarBounds;
# typedef Oracle<ScalarPenaltyL0L2, VectorBounds>   OracleScalarL0L2VectorBounds;
# typedef Oracle<VectorPenaltyL0L2, VectorBounds>   OracleVectorL0L2VectorBounds;
# typedef Oracle<ScalarPenaltyL0L1L2, VectorBounds> OracleScalarL0L1L2VectorBounds;
# typedef Oracle<VectorPenaltyL0L1L2, VectorBounds> OracleVectorL0L1L2VectorBounds;
NUM_RUNS = 50
RUN_SIZE = 10

test_that("prox finds similar solutions to linear search for L0L1L2 Bounds", {
    N = RUN_SIZE
    for (i in 1:NUM_RUNS){
        theta_opt <- as.matrix(rnorm(N), n=N, m=1)
        lows <- - as.matrix(runif(N), n=N, m=1)
        highs <-  as.matrix(runif(N), n=N, m=1)
        l0 <- as.matrix(runif(N), n=N, m=1)
        l1 <- as.matrix(runif(N), n=N, m=1)
        l2 <- as.matrix(runif(N), n=N, m=1)
        
        # Test L0L1L2 with Bounds
        oracle_vector <- gL0Learn::test_OracleVectorL0L1L2VectorBounds_prox_vec(theta_opt, l0, l1, l2, lows, highs)
        oracle_vector <- oracle_vector[, 1]
        
        test_func <- function(i) {gL0Learn::test_OracleScalarL0L1L2ScalarBounds_prox_double(theta_opt[[i]],
                                                                                            l0[[i]],
                                                                                            l1[[i]],
                                                                                            l2[[i]],
                                                                                            lows[[i]],
                                                                                            highs[[i]])}
        
        
        oracle_vector2 <- sapply(1:N, test_func)
        
        test_func_2 <- function(i) {gL0Learn.linear_search(theta_opt[[i]],
                                                           l0[[i]],
                                                           l1[[i]],
                                                           l2[[i]],
                                                           lows[[i]],
                                                           highs[[i]])}
        
        search_vector <- sapply(1:N, test_func_2)
        
        for (i in 1:N){
            info = sprintf("error found with theta=%f, l0=%f, l1=%f, l2=%f, lows=%f, highs=%f", 
                           theta_opt[[i]],
                           l0[[i]],
                           l1[[i]],
                           l2[[i]],
                           lows[[i]],
                           highs[[i]])
            expect_equal(oracle_vector[i], search_vector[i], tolerance = 1e-6, info=info)
            expect_equal(oracle_vector2[i], search_vector[i], tolerance = 1e-6, info=info)
        }
    }
})

test_that("prox finds similar solutions to linear search for L0L1L2 NoBounds", {
    N = RUN_SIZE
    for (i in 1:NUM_RUNS){
        theta_opt <- as.matrix(rnorm(N), n=N, m=1)
        l0 <- as.matrix(runif(N), n=N, m=1)
        l1 <- as.matrix(runif(N), n=N, m=1)
        l2 <- as.matrix(runif(N), n=N, m=1)
        lows <- -Inf*ones(n=N, m=1)
        highs <- Inf*ones(n=N, m=1)
        
        # Test L0L1L2 with Bounds
        oracle_vector <- gL0Learn::test_OracleVectorL0L1L2NoBounds_prox_vec(theta_opt, l0, l1, l2)
        oracle_vector <- oracle_vector[, 1]
        
        oracle_vector_b <- gL0Learn::test_OracleVectorL0L1L2VectorBounds_prox_vec(theta_opt, l0, l1, l2, lows, highs)
        oracle_vector_b <- oracle_vector_b[, 1]
        
        test_func <- function(i) {gL0Learn::test_OracleScalarL0L1L2NoBounds_prox_double(theta_opt[[i]],
                                                                                        l0[[i]],
                                                                                        l1[[i]],
                                                                                        l2[[i]])}
        
        
        oracle_vector2 <- sapply(1:N, test_func)
        
        test_func_b <- function(i) {gL0Learn::test_OracleScalarL0L1L2ScalarBounds_prox_double(theta_opt[[i]],
                                                                                        l0[[i]],
                                                                                        l1[[i]],
                                                                                        l2[[i]],
                                                                                        lows[[i]],
                                                                                        highs[[i]])}
        
        
        oracle_vector_b2 <- sapply(1:N, test_func)
        
        test_func_2 <- function(i) {gL0Learn.linear_search(theta_opt[[i]],
                                                           l0[[i]],
                                                           l1[[i]],
                                                           l2[[i]],
                                                           min(0, theta_opt[[i]]),
                                                           max(0, theta_opt[[i]]))}
        
        search_vector <- sapply(1:N, test_func_2)
        
        for (i in 1:N){
            info = sprintf("error found with theta=%f, l0=%f, l1=%f, l2=%f, lows=%f, highs=%f", 
                           theta_opt[[i]],
                           l0[[i]],
                           l1[[i]],
                           l2[[i]],
                           lows[[i]],
                           highs[[i]])
            expect_equal(oracle_vector[i], search_vector[i], tolerance = 1e-6, info=info)
            expect_equal(oracle_vector_b[i], search_vector[i], tolerance = 1e-6, info=info)
            expect_equal(oracle_vector2[i], search_vector[i], tolerance = 1e-6, info=info)
            expect_equal(oracle_vector_b2[i], search_vector[i], tolerance = 1e-6, info=info)
        }
    }
})

test_that("prox finds similar solutions to linear search for L0L2 Bounds", {
    N = RUN_SIZE
    for (i in 1:NUM_RUNS){
        theta_opt <- as.matrix(rnorm(N), n=N, m=1)
        lows <- - as.matrix(runif(N), n=N, m=1)
        highs <-  as.matrix(runif(N), n=N, m=1)
        l0 <- as.matrix(runif(N), n=N, m=1)
        l1 <- 0*ones(n=N, m=1)
        l2 <- as.matrix(runif(N), n=N, m=1)
        
        # Test L0L1L2 with Bounds
        oracle_vector <- gL0Learn::test_OracleVectorL0L2VectorBounds_prox_vec(theta_opt, l0, l2, lows, highs)
        oracle_vector <- oracle_vector[, 1]
        
        oracle_vector_l1 <- gL0Learn::test_OracleVectorL0L1L2VectorBounds_prox_vec(theta_opt, l0, l1, l2, lows, highs)
        oracle_vector_l1 <- oracle_vector_l1[, 1]
        
        test_func <- function(i) {gL0Learn::test_OracleScalarL0L2ScalarBounds_prox_double(theta_opt[[i]],
                                                                                            l0[[i]],
                                                                                            l2[[i]],
                                                                                            lows[[i]],
                                                                                            highs[[i]])}
        
        
        oracle_vector2 <- sapply(1:N, test_func)
        
        test_func_l1 <- function(i) {gL0Learn::test_OracleScalarL0L1L2ScalarBounds_prox_double(theta_opt[[i]],
                                                                                               l0[[i]],
                                                                                               l1[[i]],
                                                                                               l2[[i]],
                                                                                               lows[[i]],
                                                                                               highs[[i]])}
        
        
        oracle_vector2_l1 <- sapply(1:N, test_func_l1)
        
        test_func_2 <- function(i) {gL0Learn.linear_search(theta_opt[[i]],
                                                           l0[[i]],
                                                           l1[[i]],
                                                           l2[[i]],
                                                           lows[[i]],
                                                           highs[[i]])}
        
        search_vector <- sapply(1:N, test_func_2)
        
        for (i in 1:N){
            info = sprintf("error found with theta=%f, l0=%f, l1=%f, l2=%f, lows=%f, highs=%f", 
                           theta_opt[[i]],
                           l0[[i]],
                           l1[[i]],
                           l2[[i]],
                           lows[[i]],
                           highs[[i]])
            expect_equal(oracle_vector[i], search_vector[i], tolerance = 1e-6, info=info)
            expect_equal(oracle_vector_l1[i], search_vector[i], tolerance = 1e-6, info=info)
            expect_equal(oracle_vector2[i], search_vector[i], tolerance = 1e-6, info=info)
            expect_equal(oracle_vector2_l1[i], search_vector[i], tolerance = 1e-6, info=info)
        }
    }
})

test_that("prox finds similar solutions to linear search for L0L2 NoBounds", {
    N = RUN_SIZE
    for (i in 1:NUM_RUNS){
        theta_opt <- as.matrix(rnorm(N), n=N, m=1)
        l0 <- as.matrix(runif(N), n=N, m=1)
        l1 <- 0*ones(n=N, m=1)
        l2 <- as.matrix(runif(N), n=N, m=1)
        lows <- -Inf*ones(n=N, m=1)
        highs <- Inf*ones(n=N, m=1)
        
        # Test L0L1L2 with Bounds
        oracle_vector <- gL0Learn::test_OracleVectorL0L2NoBounds_prox_vec(theta_opt, l0, l2)
        oracle_vector <- oracle_vector[, 1]
        
        oracle_vector_l1 <- gL0Learn::test_OracleVectorL0L1L2NoBounds_prox_vec(theta_opt, l0, l1, l2)
        oracle_vector_l1 <- oracle_vector_l1[, 1]
        
        oracle_vector_b <- gL0Learn::test_OracleVectorL0L2VectorBounds_prox_vec(theta_opt, l0, l2, lows, highs)
        oracle_vector_b <- oracle_vector_b[, 1]
        
        oracle_vector_l1_b <- gL0Learn::test_OracleVectorL0L1L2VectorBounds_prox_vec(theta_opt, l0, l1, l2, lows, highs)
        oracle_vector_l1_b <- oracle_vector_l1_b[, 1]
        
        test_func <- function(i) {gL0Learn::test_OracleScalarL0L1L2NoBounds_prox_double(theta_opt[[i]],
                                                                                        l0[[i]],
                                                                                        l1[[i]],
                                                                                        l2[[i]])}
        
        
        oracle_vector2 <- sapply(1:N, test_func)
        
        test_func_nb <- function(i) {gL0Learn::test_OracleScalarL0L2NoBounds_prox_double(theta_opt[[i]],
                                                                                        l0[[i]],
                                                                                        l2[[i]])}
        
        
        oracle_vector2_nb <- sapply(1:N, test_func_nb)
        
        test_func_b <- function(i) {gL0Learn::test_OracleScalarL0L2ScalarBounds_prox_double(theta_opt[[i]],
                                                                                             l0[[i]],
                                                                                             l2[[i]],
                                                                                             lows[[i]],
                                                                                             highs[[i]])}
        
        
        oracle_vector2_b <- sapply(1:N, test_func_b)
        
        test_func_b_l1 <- function(i) {gL0Learn::test_OracleScalarL0L1L2ScalarBounds_prox_double(theta_opt[[i]],
                                                                                              l0[[i]],
                                                                                              l1[[i]],
                                                                                              l2[[i]],
                                                                                              lows[[i]],
                                                                                              highs[[i]])}
        
        
        oracle_vector2_b_l1 <- sapply(1:N, test_func_b_l1)
        
        test_func_2 <- function(i) {gL0Learn.linear_search(theta_opt[[i]],
                                                           l0[[i]],
                                                           l1[[i]],
                                                           l2[[i]],
                                                           min(0, theta_opt[[i]]),
                                                           max(0, theta_opt[[i]]))}
        
        search_vector <- sapply(1:N, test_func_2)
        
        for (i in 1:N){
            info = sprintf("error found with theta=%f, l0=%f, l1=%f, l2=%f, lows=%f, highs=%f", 
                           theta_opt[[i]],
                           l0[[i]],
                           l1[[i]],
                           l2[[i]],
                           lows[[i]],
                           highs[[i]])
            expect_equal(oracle_vector[i], search_vector[i], tolerance = 1e-6, info=info)
            expect_equal(oracle_vector_l1[i], search_vector[i], tolerance = 1e-6, info=info)
            expect_equal(oracle_vector_l1_b[i], search_vector[i], tolerance = 1e-6, info=info)
            expect_equal(oracle_vector_b[i], search_vector[i], tolerance = 1e-6, info=info)
            expect_equal(oracle_vector2[i], search_vector[i], tolerance = 1e-6, info=info)
            expect_equal(oracle_vector2_nb[i], search_vector[i], tolerance = 1e-6, info=info)
            expect_equal(oracle_vector2_b[i], search_vector[i], tolerance = 1e-6, info=info)
            expect_equal(oracle_vector2_b_l1[i], search_vector[i], tolerance = 1e-6, info=info)
        }
    }
})
