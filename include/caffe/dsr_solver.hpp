#ifndef CAFFE_DSR_SOLVERS_HPP_
#define CAFFE_DSR_SOLVERS_HPP_
#include "caffe/sgd_solvers.hpp"
#include <json/writer.h>


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
  virtual void ComputeUpdateValue(int param_id, Dtype rate);
  virtual Dtype GetParamMomentum(int param_id);
  
  vector<shared_ptr<Blob<Dtype> > > line_ ,abs_line_ , path_, ratio_ , prev_ ,diff_ , update_;
  Json::Value record_;
  DISABLE_COPY_AND_ASSIGN(DSRSolver);
};



}


#endif
