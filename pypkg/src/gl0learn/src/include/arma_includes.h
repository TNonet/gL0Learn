#ifndef INCLUDES_H
#define INCLUDES_H

#include <exception>
#include <stdexcept>
#include <utility>

#include <armadillo>
#include <carma>

#define COUT std::cout

// A type that should be translated to a standard Python exception
class PyException : public std::exception {
public:
  explicit PyException(const char *m) : message{m} {}
  const char *what() const noexcept override { return message.c_str(); }

private:
  std::string message;
};

#define STOP throw PyException

#endif // INCLUDES_H