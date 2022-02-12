#ifndef H_ACTIVE_SET
#define H_ACTIVE_SET
#include <tuple>
#include <vector>
#include "arma_includes.h"


typedef std::tuple<arma::uword, arma::uword> coordinate;
typedef std::vector<coordinate> coordinate_vector;

template <typename T>
std::ostream& operator<<(std::ostream& stream, const std::tuple<T, T>& c)
{
    return stream << "{" << std::get<0>(c) << ", " << std::get<1>(c) << "}";
}

template <typename T>
std::ostream& operator<<(std::ostream& stream, const std::vector<std::tuple<T, T>>& c)
{
    for (auto c_i: c){
        stream << c_i << "\n";
    }
    
    return stream;
}

inline coordinate inc(const coordinate c, const arma::uword p){
    const double i = std::get<0>(c);
    const double j = std::get<1>(c);
    if (std::get<1>(c) < p - 1){
        return {i, j+1};
    } else if (std::get<0>(c) < p - 1){
        return {i+1, 0};
    } else {
        COUT << "Cannot increment coordinate (" << i << ", " << j << ")as it is already at maximium";
        STOP("Error in coordinate inc");
    }
};

inline coordinate_vector upper_triangle_coordinate_vector(const arma::uword p){
    coordinate_vector coord_vec;
    coord_vec.reserve(p*(p-1));
    for (arma::uword i=0; i<p; i++){
        for (arma::uword j=i+1; j<p; j++){
            coord_vec.push_back({i, j});
        }
    }
    
    return coord_vec;
}

template <typename T>
inline coordinate_vector union_of_correlated_features(const T& x, const double correlation_threshold){
    /*
     *  Returns the coordinates of the upper triangle of xtx where xtx[i,j] > correlation_threshold
     *  
     */
    
    const auto p = x.n_cols;
    coordinate_vector active_set = coordinate_vector();
    active_set.reserve(p*(p-1));

    for (auto i = 0; i < p - 1; ++i){
        const arma::vec xxt_i = x.cols(i+1, p-1).t()*x.col(i);
        const arma::uvec highly_correlated_xxt_i_indicies = arma::find(arma::abs(xxt_i) > correlation_threshold);
        
        arma::uvec::const_iterator it = highly_correlated_xxt_i_indicies.begin();
        const arma::uvec::const_iterator it_end = highly_correlated_xxt_i_indicies.end();
        
        for(; it != it_end; ++it){
            active_set.push_back({i,(*it) + i + 1});
        }
    }
    
    return active_set;
}

inline arma::umat unravel_ut_indices(const arma::uvec& transposed_indices,
                                     const arma::uword p){
    /*
     * Unravels upper triangular indices of an matrix into lexigraphically 
     *  sorted coordinates of a matrix.
     * 
     * Requies `transposed_indices` be the indices of the of the transposed 
     *  matrix. For example, instead of;
     *      indices <- find(x > 1)
     *  It shoudl be:
     *      indices <- find(x.t() > 1).
     * 
     * Given the indices of a square p by p matrix:
     *   +-----+--------+-----+----------+
     *   | 0 X | p      | ... | p(p-1)   |
     *   +-----+--------+-----+----------+
     *   | -   | p+1  X | ... | p(p-1)+1 |
     *   +-----+--------+-----+----------+
     *   | ... | ...    | ... | ...      |
     *   +-----+--------+-----+----------+
     *   | --  | ---    | ... | p*p-1  X |
     *   +-----+--------+-----+----------+
     *   
     *   To convert to coordinates (i, j) we notice that:
     *      i [row] = index % p
     *      j [col] = index // p (floor divide)
     *      
     *  Note, if you transpose a matrix using .t(), the indices flip!
     *  +--------+----------+-----+--------+
     *  | 0      | 1        | ... | p-1    |
     *  +--------+----------+-----+--------+
     *  | p      | p+1      | ... | 2(p-1) |
     *  +--------+----------+-----+--------+
     *  | ...    | ...      | ... | ...    |
     *  +--------+----------+-----+--------+
     *  | p(p-1) | p(p-1)+1 | ... | p*p-1  |
     *  +--------+----------+-----+--------+
     *      
     */
    arma::umat coords(transposed_indices.n_elem, 2);
    
    coords.col(0) = arma::floor(transposed_indices/p); 
    coords.col(1) = transposed_indices - coords.col(0)*p;

    return coords;
}

bool inline check_is_coordinate_subset(const arma::umat& larger_coord_set,
                                       const arma::umat& smaller_coord_set){
    /*
     *  Determines if `smaller_coord_set` is contained in `larger_coord_set`.
     *  
     *  Both `*_coord_set` must be sorted coordinate_matrices. For example, 
     *  from:
     *   `union_of_correlated_features2`
     *   `coordinate_matrix_from_vector`
     */
    
    if (smaller_coord_set.is_empty()){return true;}
    if (larger_coord_set.is_empty()){return false;}
    
    const auto max_col = std::max(larger_coord_set.col(1).max(),
                                  smaller_coord_set.col(1).max()) + 1;
                                   
    const arma::uvec larger_indices = larger_coord_set.col(0)*max_col + larger_coord_set.col(1);
    const arma::uvec smaller_indices = smaller_coord_set.col(0)*max_col + smaller_coord_set.col(1);
    
    return std::includes(larger_indices.begin(),
                         larger_indices.end(),
                         smaller_indices.begin(),
                         smaller_indices.end());

}

bool inline check_coordinate_matrix(const arma::umat& coords_ma,
                                    const bool for_order = true,
                                    const bool for_upper_triangle = true){
    /*
     *  Checks `coords_ma` to ensure:
     *      if `for_order` is true, that the ordering is lexicographically 
     *       correct. This also checks for duplicates (which are not allowed)
     *      if `for_upper_triangle` is true, that the values are limited to the 
     *       upper triangle of a matrix.
     */
    
    if (coords_ma.is_empty()){
        return true;
    }
    
    bool check = true;
    
    if (for_order){
        const auto max_col = coords_ma.col(1).max() + 1;
        const arma::uvec indices = coords_ma.col(0)*max_col + coords_ma.col(1);
        check = check & indices.is_sorted("strictascend");
    }
    
    if (for_upper_triangle){
        check = check & arma::all(coords_ma.col(0) < coords_ma.col(1));
    }
    
    return check;
}

template <typename T>
inline arma::umat union_of_correlated_features2(const T& x, const double correlation_threshold){
    /*
     *  Returns the coordinates of the upper triangle, not including diagonal
     *   of xtx where xtx[i,j] > correlation_threshold
     *  
     *  Note, because we are using armadillo logic which is column major, 
     *   we have to flip our coordinates to make this work.
     */
    const auto p = x.n_cols;
    arma::mat upper_triangle_indicator(p, p, arma::fill::zeros);
    upper_triangle_indicator.elem(arma::trimatu_ind(arma::size(upper_triangle_indicator), 1)).fill(1);
    const arma::uvec highly_correlated_indicies = arma::find((arma::abs(x.t()*x)%upper_triangle_indicator).t() > correlation_threshold);
    return unravel_ut_indices(highly_correlated_indicies, p);
}

inline coordinate_vector coordinate_vector_from_matrix(const arma::umat& coords_ma){
    const arma::uword n = coords_ma.n_rows;
    
    coordinate_vector coords_vec;
    coords_vec.reserve(n);
    
    for (arma::uword row_index = 0; row_index < n; ++row_index){
        arma::umat::const_row_iterator it_row_begin = coords_ma.begin_row(row_index);
        const auto i = (*it_row_begin);
        ++it_row_begin;
        const auto j = (*it_row_begin);
        coords_vec.push_back({i, j});
    }
    return coords_vec;
}

inline arma::umat coordinate_matrix_from_vector(const coordinate_vector& coord_vec){
    const auto n = coord_vec.size();
    arma::umat coords_ma(n, 2);
    
    auto it = coord_vec.begin();
    
    for (arma::uword row_index = 0; row_index < n; ++row_index, ++it){
        arma::umat::row_iterator it_row_begin = coords_ma.begin_row(row_index);
        (*it_row_begin) = std::get<0>(*it);
        ++it_row_begin;
        (*it_row_begin) = std::get<1>(*it);
    }
    
    return coords_ma;
}

template <typename T>
inline coordinate_vector support_from_active_set(const T& x, const coordinate_vector active_set){
    
    coordinate_vector support = coordinate_vector(active_set.size()); 
    for (auto ij: active_set){
        const double i = std::get<0>(ij);
        const double j = std::get<1>(ij);
        
        if (x(i, j) != 0){
            support.push_back(ij);
        }
        
    }
    support.shrink_to_fit();
    return support;
}

template <typename T>
std::vector<T> sorted_vector_difference(const std::vector<T>& larger, const std::vector<T>& smaller){
    /*
     *  Returns all items in `larger` that aren't in `smaller`.
     *  Assumes both larger and smaller are sorted in the same fashion.
     */
    auto larger_size = larger.size();
    auto smaller_size = smaller.size();
    std::vector<T> difference  = {};
    difference.reserve(larger_size - smaller_size);
    
    auto alpha = 0;
    for (auto const &ij: larger){
        if ((alpha < smaller_size) && (ij == smaller[alpha])){
            ++alpha;
        } else {
            difference.push_back(ij);
        }
    }
    return difference;
}

template <typename T>
std::vector<T> sorted_vector_difference2(const std::vector<T>& larger, const std::vector<T>& smaller){
    /*
     *  Returns all items in `larger` that aren't in `smaller`.
     *  Assumes both larger and smaller are sorted in the same fashion.
     */
    auto larger_size = larger.size();
    auto smaller_size = smaller.size();
    std::vector<T> difference  = {};
    difference.reserve(larger_size - smaller_size);
    
    std::set_difference(larger.begin(), larger.end(),
                        smaller.begin(), smaller.end(),
                        std::inserter(difference, difference.begin()));
    
    return difference;
}

template <typename T>
std::vector<T> insert_sorted_vector_into_sorted_vector(const std::vector<T>& x1, const std::vector<T>& x2){
    /*
     *  Combines `base` and `by`
     *  Assumes both larger and smaller are sorted in the same fashion.
     */
    std::vector<T> merged;
    merged.reserve(x1.size() + x2.size());
    std::merge(x1.begin(), x1.end(),
               x2.begin(), x2.end(),
               std::back_inserter(merged));
    
    return merged;
}




#endif // H_ACTIVE_SET