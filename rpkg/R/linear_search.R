# import C++ compiled code
#' @importFrom pracma linspace
#' 
#' @title gL0Learn.linear_search
#'
#' @description Linearly searches a linspace from `lows` to `highs` to find min 
#' value of |theta_opt - x| + l0|x|_0 + l1|x|_1 + + l2|x|_2
#' @param x 

#' @examples
#' MISSING
#' @export
NULL
gL0Learn.linear_search <- function(theta_opt, l0=0, l1=0, l2=0, lows=-1, highs=+1, atol=1e-6){
    N <- (highs - lows) / atol
    x <- linspace(lows, highs, N)
    
    if (!(0. %in% x)){
        last_negative_number_index = max(which(x < 0))
        x <- append(x, 0., after=last_negative_number_index)
    }
    f_x <- 0.5*(theta_opt-x)**2 + (x!=0.)*l0 + abs(x)*l1 + x*x*l2
    
    return(x[which.min(f_x)])
}

# import C++ compiled code
#' @importFrom pracma linspace
#' 
#' @title gL0Learn.linear_plot
#'
#' @description Linearly searches a linspace from `lows` to `highs` to find min 
#' value of |theta_opt - x| + l0|x|_0 + l1|x|_1 + + l2|x|_2
#' @param x 

#' @examples
#' MISSING
#' @export
NULL
gL0Learn.linear_plot<- function(theta_opt, l0=0, l1=0, l2=0, lows=-1, highs=+1, atol=1e-3){
    N <- (highs - lows) / atol
    x <- linspace(lows, highs, N)
    
    if (!(0. %in% x)){
        last_negative_number_index = max(which(x < 0))
        x <- append(x, 0., after=last_negative_number_index)
    }
    f_x <- 0.5*(theta_opt-x)**2 + (x!=0.)*l0 + abs(x)*l1 + x*x*l2
    
    data <- data.frame(x,f_x)
    plot(data, aes(x=x, y=f_x)) + geom_line()
}