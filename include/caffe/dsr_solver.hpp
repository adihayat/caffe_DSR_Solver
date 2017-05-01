#ifndef CAFFE_DSR_SOLVERS_HPP_
#define CAFFE_DSR_SOLVERS_HPP_
#include "caffe/sgd_solvers.hpp"

namespace caffe {

template <typename Dtype>
class DSRSolver : public SGDSolver<Dtype>
{
public:
  explicit DSRSolver(const SolverParameter& param)
      : SGDSolver<Dtype>(param) { AllocateLinePath(); }
  explicit DSRSolver(const string& param_file)
      : SGDSolver<Dtype>(param_file) { AllocateLinePath(); }
  virtual inline const char* type() const { return "DSR"; }

protected:
  
  void AllocateLinePath();
  virtual void ApplyUpdate();
  virtual void ComputeUpdateValue(int param_id,Dtype rate);

  vector<shared_ptr<Blob<Dtype> > > line_;
  vector<Dtype> path_;
  vector<Dtype> ratio_;
  DISABLE_COPY_AND_ASSIGN(DSRSolver);
};



}


#endif
