library("gL0Learn")

n <- 1000
p <- 100
data <- gL0Learn.generate_synthetic(n,
  p,
  "KR1",
  normalize = "covariance",
  seed = 1,
  rho = 0.5
)

# test_that("gL0Learn.fit fails with bad X matrix", {
#     expect_error(gL0Learn.gfit(rnorm(100), rnorm(100)))
#     expect_error(gL0Learn.gfit(matrix(rnorm(100), 100, 1), rnorm(100)))
# })
#
# test_that("gL0Learn.fit fails with bad theta_init matrix", {
#     fail()
# })
