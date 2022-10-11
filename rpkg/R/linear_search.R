#' @importFrom pracma linspace
#' @title Linear grid search minimization of gL0Learn regularized loss.
#' @description Linearly searches a linspace from `lows` to `highs` to find arg
#' min value of x for:
#'  L(theta_opt, x) = (|theta_opt - x|)**2 + l0|x|_0 + l1|x|_1 + + l2|x|_2
#' We ensure that 0 is included in search space.
#' @param theta_opt Optimal theta value that we are minimizing approximation off
#' @param l0 L0 regularization penalty. Must be a non-negative scalar
#' @param l1 L1 regularization penalty. Must be a non-negative scalar
#' @param l2 L2 regularization penalty. Must be a non-negative scalar
#' @param lows lower bound for x. Can be 0 but `highs` must not be 0 at the
#' same time
#' @param highs upper bound for x. Can be 0 but `lows` must not be 0 at the same time
#' @param atol Step size between each successive point in linspace
#' @param return_all boolean flag whether or not to return L(x, theta) for each
#' value in the linspace or just the arg min.
#' @export
gL0Learn.linear_search <- function(theta_opt, # nolint
                                   l0 = 0,
                                   l1 = 0,
                                   l2 = 0,
                                   lows = -1,
                                   highs = +1,
                                   atol = 1e-6,
                                   return_all = FALSE) {
  n <- (highs - lows) / atol
  x <- linspace(lows, highs, n)

  if (!(0. %in% x)) {
    last_negative_number_index <- max(which(x < 0))
    x <- append(x, 0., after = last_negative_number_index)
  }
  f_x <- (0.5 * (theta_opt - x)**2
    + l0 * (x != 0.)
    + l1 * abs(x)
    + l2 * x**2)

  if (return_all) {
    return(list(x = x, f_x = f_x))
  } else {
    return(x[which.min(f_x)])
  }
}

#' @importFrom pracma linspace
#' @importFrom ggplot2 geom_line aes
#' @title Linear Plot of gL0Learn regularized loss.
#' @description Plots value of L(theta_opt, x) over a the linspace from `lows` 
#' to `highs` to display shape of L for:
#'  L(theta_opt, x) = (|theta_opt - x|)**2 + l0|x|_0 + l1|x|_1 + + l2|x|_2
#' @param theta_opt See `gL0Learn.linear_search`
#' @param l0 See `gL0Learn.linear_search`
#' @param l1 See `gL0Learn.linear_search`
#' @param l2 See `gL0Learn.linear_search`
#' @param lows See `gL0Learn.linear_search`
#' @param highs See `gL0Learn.linear_search`
#' @param atol See `gL0Learn.linear_search`
#' @export
gL0Learn.linear_plot <- function(theta_opt, # nolint
                                 l0 = 0,
                                 l1 = 0,
                                 l2 = 0,
                                 lows = -1,
                                 highs = +1,
                                 atol = 1e-3) {
  data <- gL0Learn.linear_search(
    theta_opt = theta_opt,
    l0 = l0,
    l1 = l1,
    l2 = l2,
    lows = lows,
    highs = highs,
    atol = atol,
    return_all = TRUE
  )
  data <- data.frame(x = data$x, y = data$f_x)
  plot(data) + geom_line()
}
