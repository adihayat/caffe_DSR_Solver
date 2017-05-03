#include "caffe/dsr_solver.hpp"
#include <cmath>
#include <json/writer.h>
#include <iostream>
#include <string>

namespace caffe {

template <typename Dtype>
void DSRSolver<Dtype>::AllocateLinePath() {
  // Initialize the history
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  line_.clear();
  path_.clear();
  ratio_.clear();
  for (int i = 0; i < net_params.size(); ++i) {
    const vector<int>& shape = net_params[i]->shape();
    line_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    path_.push_back(0);
    ratio_.push_back(0);
  }
}



template <typename Dtype>
void DSRSolver<Dtype>::ApplyUpdate() {
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  Dtype dsr_decay = this->param_.dsr_decay();
  Json::Value & record = this->record_;
  Dtype sum_ratio = 0;
  Dtype dsr_decay_comp = 1 - dsr_decay;
  int   param_count = 0;

  for (int param_id = 0; param_id < net_params.size(); ++param_id) {
      caffe_cpu_axpby(net_params[param_id]->count(), dsr_decay_comp,
                net_params[param_id]->cpu_diff(), dsr_decay,
                line_[param_id]->mutable_cpu_data());
      
      Dtype current_diff_norm = 0; 
      caffe_cpu_gemv(CblasNoTrans,1,net_params[param_id]->count(), Dtype(1.0),
                net_params[param_id]->cpu_diff(), 
                net_params[param_id]->cpu_diff(), 
                Dtype(0),
                &(current_diff_norm));

      path_[param_id] = path_[param_id] * dsr_decay + std::sqrt(current_diff_norm) * (1-dsr_decay_comp);

      Dtype line_norm = 0;
      caffe_cpu_gemv(CblasNoTrans,1,net_params[param_id]->count(), Dtype(1.0),
                line_[param_id]->cpu_data(), 
                line_[param_id]->cpu_data(), 
                Dtype(0),
                &(line_norm));

      line_norm = std::sqrt(line_norm);


      ratio_[param_id] = line_norm/(path_[param_id] + this->param_.dsr_eps());
            
      if (!record.isMember(std::to_string(param_id))) {
          record[std::to_string(param_id)]["ratio"] = Json::Value(Json::arrayValue);
          record[std::to_string(param_id)]["line"] = Json::Value(Json::arrayValue);
          record[std::to_string(param_id)]["path"] = Json::Value(Json::arrayValue);
          record[std::to_string(param_id)]["lr_fix"] = Json::Value(Json::arrayValue);
          record[std::to_string(param_id)]["size"] = net_params[param_id]->count();
      }

      record[std::to_string(param_id)]["ratio"].append(ratio_[param_id]);
      record[std::to_string(param_id)]["line"].append(line_norm);
      record[std::to_string(param_id)]["path"].append(path_[param_id]);
     
      if (net_params[param_id]->count() >= this->param_.dsr_min_dim()) { 
        param_count += 1;
        sum_ratio += ratio_[param_id];
      }

  }
  
  if ((this->iter_ % 2000) == 0) {
      std::ofstream out("ratio.json");
      out << record;
      out.close();
  }

  this->mean_ratio_ = (this->mean_ratio_ * dsr_decay + sum_ratio / param_count) / (1 + dsr_decay);

  SGDSolver<Dtype>::ApplyUpdate();
}

template <typename Dtype>
Dtype DSRSolver<Dtype>::GetParamLr(int param_id)
{
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  const vector<float>& net_params_lr = this->net_->params_lr();
  Json::Value & record = this->record_;
  if (net_params[param_id]->count() < this->param_.dsr_min_dim()) {
      record[std::to_string(param_id)]["lr_fix"].append(1);
      return net_params_lr[param_id];
  }

  float norm_ratio = ratio_[param_id] / this->mean_ratio_;
  norm_ratio = norm_ratio < this->param_.dsr_min_ratio() ? this->param_.dsr_min_ratio() : 
               norm_ratio > this->param_.dsr_max_ratio() ? this->param_.dsr_max_ratio() : norm_ratio;
  record[std::to_string(param_id)]["lr_fix"].append(norm_ratio);

  return net_params_lr[param_id] * norm_ratio;
}

INSTANTIATE_CLASS(DSRSolver);
REGISTER_SOLVER_CLASS(DSR);


}
