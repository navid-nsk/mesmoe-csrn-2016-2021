#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

// Utility for checking CUDA errors
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Forward declarations of CUDA kernel functions
torch::Tensor compute_gradient_cuda(
    torch::Tensor population,
    torch::Tensor da_coordinates,
    int max_neighbors
);

std::vector<torch::Tensor> compute_gradient_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor population,
    torch::Tensor da_coordinates,
    int max_neighbors
);

torch::Tensor compute_divergence_cuda(
    torch::Tensor vector_field,
    torch::Tensor da_coordinates,
    int max_neighbors
);

std::vector<torch::Tensor> compute_divergence_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor vector_field,
    torch::Tensor da_coordinates,
    int max_neighbors
);

std::vector<torch::Tensor> compute_pde_terms_cuda(
    torch::Tensor population,
    torch::Tensor gradient,
    torch::Tensor diffusion,
    torch::Tensor amplification,
    torch::Tensor velocity,
    torch::Tensor alpha,
    torch::Tensor beta,
    torch::Tensor gamma,
    torch::Tensor laplacian,
    torch::Tensor velocity_div,
    float sigma_squared,
    float K,
    float pop_scale_factor
);

std::vector<torch::Tensor> compute_pde_terms_backward_cuda(
    torch::Tensor grad_diffusion_term,
    torch::Tensor grad_directed_flow,
    torch::Tensor grad_mean_reversion,
    torch::Tensor grad_gradient_penalty,
    torch::Tensor grad_advection_stab,
    torch::Tensor population,
    torch::Tensor gradient,
    torch::Tensor diffusion,
    torch::Tensor amplification,
    torch::Tensor velocity,
    torch::Tensor alpha,
    torch::Tensor beta,
    torch::Tensor gamma,
    torch::Tensor laplacian,
    torch::Tensor velocity_div,
    float sigma_squared,
    float K,
    float pop_scale_factor
);

torch::Tensor rk4_step_cuda(
    torch::Tensor population,
    torch::Tensor k1,
    torch::Tensor k2,
    torch::Tensor k3,
    torch::Tensor k4,
    float dt
);

std::vector<torch::Tensor> rk4_step_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor population,
    torch::Tensor k1,
    torch::Tensor k2,
    torch::Tensor k3,
    torch::Tensor k4,
    float dt,
    bool is_log_space
);

// Helper function to clip gradient norms
torch::Tensor clip_gradient_norm(torch::Tensor grad, float max_norm) {
    float grad_norm = grad.norm().item<float>();
    
    if (grad_norm > max_norm) {
        // std::cout << "Clipping gradient norm from " << grad_norm << " to " << max_norm << std::endl;
        return grad * (max_norm / grad_norm);
    }
    
    return grad;
}

//========================================================================
// Custom autograd functions for PyTorch
//========================================================================

// Gradient computation with autograd support
class GradientFunction : public torch::autograd::Function<GradientFunction> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor population,
        torch::Tensor da_coordinates,
        int max_neighbors
    ) {
        // Save inputs for backward
        ctx->save_for_backward({population, da_coordinates});
        ctx->saved_data["max_neighbors"] = max_neighbors;
        
        // Call the forward implementation
        auto result = compute_gradient_cuda(population, da_coordinates, max_neighbors);
        
        // Debug output to verify tensor data ranges
        // std::cout << "Gradient forward result range: [" 
        //           << result.min().item<float>() << ", " 
        //           << result.max().item<float>() << "]" << std::endl;
                  
        return result;
    }
    
    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        // Retrieve saved tensors and options
        auto saved = ctx->get_saved_variables();
        auto population = saved[0];
        auto da_coordinates = saved[1];
        int max_neighbors = ctx->saved_data["max_neighbors"].toInt();
        
        // Ensure grad_output is contiguous
        auto grad_output = grad_outputs[0].contiguous();
        
        // Check for NaN or extreme values
        bool has_nan = grad_output.isnan().any().item<bool>();
        bool has_inf = grad_output.isinf().any().item<bool>();
        
        if (has_nan || has_inf) {
            std::cout << "WARNING: Gradient backward input contains NaN/Inf! Fixing..." << std::endl;
            grad_output = torch::nan_to_num(grad_output, 0.0, 1.0, -1.0);
        }
        
        // Clip gradient norm to prevent explosion
        grad_output = clip_gradient_norm(grad_output, 100.0);
        
        // Debug output to verify gradient flow
        // std::cout << "Gradient backward input norm: " 
        //           << grad_output.norm().item<float>() << std::endl;
        
        // Call the backward implementation
        auto grads = compute_gradient_backward_cuda(
            grad_output, population, da_coordinates, max_neighbors
        );
        
        // Clip gradient values to prevent explosion
        if (grads[0].defined()) {
            grads[0] = clip_gradient_norm(grads[0], 1000.0);
            
            // std::cout << "Population gradient norm: " 
            //           << grads[0].norm().item<float>() << std::endl;
        }
        
        // Return gradients for each input: population, da_coordinates, max_neighbors
        // Coordinates don't need gradients, and max_neighbors is not a tensor
        return {grads[0], torch::Tensor(), torch::Tensor()};
    }
};

// Divergence computation with autograd support
class DivergenceFunction : public torch::autograd::Function<DivergenceFunction> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor vector_field,
        torch::Tensor da_coordinates,
        int max_neighbors
    ) {
        // Save inputs for backward
        ctx->save_for_backward({vector_field, da_coordinates});
        ctx->saved_data["max_neighbors"] = max_neighbors;
        
        // Call the forward implementation
        return compute_divergence_cuda(vector_field, da_coordinates, max_neighbors);
    }
    
    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        // Retrieve saved tensors and options
        auto saved = ctx->get_saved_variables();
        auto vector_field = saved[0];
        auto da_coordinates = saved[1];
        int max_neighbors = ctx->saved_data["max_neighbors"].toInt();
        
        // Ensure grad_output is contiguous
        auto grad_output = grad_outputs[0].contiguous();
        
        // Check for NaN values in grad_output
        bool has_nan = grad_output.isnan().any().item<bool>();
        bool has_inf = grad_output.isinf().any().item<bool>();
        
        if (has_nan || has_inf) {
            std::cout << "WARNING: Divergence grad_output contains NaN/Inf! Fixing..." << std::endl;
            // Replace NaNs/Infs with zeros to prevent propagation
            grad_output = torch::nan_to_num(grad_output, 0.0, 1.0, -1.0);
        }
        
        // Clip gradient norm
        grad_output = clip_gradient_norm(grad_output, 100.0);

        // Call the backward implementation
        auto grads = compute_divergence_backward_cuda(
            grad_output, vector_field, da_coordinates, max_neighbors
        );
        
        // Clip gradients
        if (grads[0].defined()) {
            grads[0] = clip_gradient_norm(grads[0], 1000.0);
        }
        
        // Return gradients for each input: vector_field, da_coordinates, max_neighbors
        return {grads[0], torch::Tensor(), torch::Tensor()};
    }
};

// PDE terms computation with autograd support
class PDETermsFunction : public torch::autograd::Function<PDETermsFunction> {
public:
    static std::vector<torch::Tensor> forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor population,
        torch::Tensor gradient,
        torch::Tensor diffusion,
        torch::Tensor amplification,
        torch::Tensor velocity,
        torch::Tensor alpha,
        torch::Tensor beta,
        torch::Tensor gamma,
        torch::Tensor laplacian,
        torch::Tensor velocity_div,
        double sigma_squared,
        double K,
        double pop_scale_factor
    ) {
        // Save inputs for backward
        ctx->save_for_backward({
            population, gradient, diffusion, amplification, velocity,
            alpha, beta, gamma, laplacian, velocity_div
        });
        ctx->saved_data["sigma_squared"] = sigma_squared;
        ctx->saved_data["K"] = K;
        ctx->saved_data["pop_scale_factor"] = pop_scale_factor;
        
        // Call the forward implementation
        auto results = compute_pde_terms_cuda(
            population, gradient, diffusion, amplification, velocity,
            alpha, beta, gamma, laplacian, velocity_div, 
            sigma_squared, K, pop_scale_factor
        );
        
        // Debug output for each term
        // for (size_t i = 0; i < results.size(); ++i) {
        //     std::string term_names[] = {"diffusion_term", "directed_flow", 
        //                                "mean_reversion", "gradient_penalty", 
        //                                "advection_stabilization"};
        //     std::cout << term_names[i] << " range: [" 
        //              << results[i].min().item<float>() << ", " 
        //              << results[i].max().item<float>() << "]" << std::endl;
        // }
        
        return results;
    }
    
    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        // Retrieve saved tensors and options
        auto saved = ctx->get_saved_variables();
        auto population = saved[0];
        auto gradient = saved[1];
        auto diffusion = saved[2];
        auto amplification = saved[3];
        auto velocity = saved[4];
        auto alpha = saved[5];
        auto beta = saved[6];
        auto gamma = saved[7];
        auto laplacian = saved[8];
        auto velocity_div = saved[9];
        
        double sigma_squared = ctx->saved_data["sigma_squared"].toDouble();
        double K = ctx->saved_data["K"].toDouble();
        double pop_scale_factor = ctx->saved_data["pop_scale_factor"].toDouble();
        
        // Debug output for gradient inputs
        // std::cout << "PDE Terms backward input:";
        float max_norm = 0.0f;
        for (size_t i = 0; i < grad_outputs.size(); ++i) {
            if (grad_outputs[i].defined()) {
                float norm = grad_outputs[i].norm().item<float>();
                // std::cout << " grad[" << i << "] norm=" << norm;
                max_norm = fmax(max_norm, norm);
            } else {
                std::cout << " grad[" << i << "] undefined";
            }
        }
        // std::cout << std::endl;
        
        // Ensure all grad_outputs tensors are valid and contiguous
        std::vector<torch::Tensor> processed_grads;
        
        // Max gradient norm allowed
        float clip_norm = 1000.0f;
        
        // If max norm is very large, use a more aggressive clipping
        if (max_norm > 1e6) {
            clip_norm = 100.0f;
        } else if (max_norm > 1e3) {
            clip_norm = 500.0f;
        }
        
        for (size_t i = 0; i < grad_outputs.size(); ++i) {
            if (grad_outputs[i].defined()) {
                auto grad = grad_outputs[i].contiguous();
                
                // Check for NaN/Inf values and replace them
                if (grad.isnan().any().item<bool>() || grad.isinf().any().item<bool>()) {
                    std::cout << "WARNING: NaN/Inf detected in PDE term " << i << " gradient!" << std::endl;
                    grad = torch::nan_to_num(grad, 0.0, 1.0, -1.0);
                }
                
                // Clip norm
                grad = clip_gradient_norm(grad, clip_norm);
                
                processed_grads.push_back(grad);
            } else {
                // Create zero tensors of the right shape if gradients are undefined
                processed_grads.push_back(torch::zeros_like(population));
            }
        }
        
        // Ensure we have all 5 gradient outputs
        while (processed_grads.size() < 5) {
            processed_grads.push_back(torch::zeros_like(population));
        }

        // Call the backward implementation
        auto grads = compute_pde_terms_backward_cuda(
            processed_grads[0], processed_grads[1], processed_grads[2], 
            processed_grads[3], processed_grads[4],
            population, gradient, diffusion, amplification, velocity,
            alpha, beta, gamma, laplacian, velocity_div,
            sigma_squared, K, pop_scale_factor
        );
        
        // Debug output for output gradients and clip large values
        // std::cout << "PDE Terms gradient outputs:";
        // for (size_t i = 0; i < grads.size(); ++i) {
        //     if (grads[i].defined()) {
        //         // Check for NaN/Inf and clip
        //         if (grads[i].isnan().any().item<bool>() || grads[i].isinf().any().item<bool>()) {
        //             std::cout << " grad[" << i << "] contains NaN/Inf!";
        //             grads[i] = torch::nan_to_num(grads[i], 0.0, 1.0, -1.0);
        //         }
                
        //         // Clip gradient norm
        //         grads[i] = clip_gradient_norm(grads[i], 10000.0);
                
        //         std::cout << " grad[" << i << "] norm=" << grads[i].norm().item<float>();
        //     } else {
        //         std::cout << " grad[" << i << "] undefined";
        //     }
        // }
        // std::cout << std::endl;
        
        // Return gradients for each input (non-tensor inputs get None)
        return {
            grads[0], grads[1], grads[2], grads[3], grads[4],
            grads[5], grads[6], grads[7], grads[8], grads[9],
            torch::Tensor(), torch::Tensor(), torch::Tensor()
        };
    }
};

// RK4 step computation with autograd support
class RK4StepFunction : public torch::autograd::Function<RK4StepFunction> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext *ctx,
        torch::Tensor population,
        torch::Tensor k1,
        torch::Tensor k2,
        torch::Tensor k3,
        torch::Tensor k4,
        double dt,
        bool is_log_space
    ) {
        // Save inputs for backward
        ctx->save_for_backward({population, k1, k2, k3, k4});
        ctx->saved_data["dt"] = dt;
        ctx->saved_data["is_log_space"] = is_log_space;
        
        // Call the forward implementation
        auto result = rk4_step_cuda(population, k1, k2, k3, k4, dt);
        
        // Debug output to verify tensor values
        // std::cout << "RK4 step result range: [" 
        //           << result.min().item<float>() << ", " 
        //           << result.max().item<float>() << "]" << std::endl;
                  
        // Check if result is identical to input
        bool identical = torch::allclose(result, population, 1e-5, 1e-5);
        if (identical) {
            std::cout << "WARNING: RK4 result identical to input! Adding perturbation." << std::endl;
            
            // Add a small perturbation to ensure gradient flow
            auto perturbation = population * 0.001;
            // Ensure perturbation isn't zero where population is zero
            perturbation = perturbation + 0.0001;
            result = result + perturbation;
        }
        
        return result;
    }
    
    static torch::autograd::tensor_list backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::tensor_list grad_outputs
    ) {
        // Retrieve saved tensors and options
        auto saved = ctx->get_saved_variables();
        auto population = saved[0];
        auto k1 = saved[1];
        auto k2 = saved[2];
        auto k3 = saved[3];
        auto k4 = saved[4];
        
        double dt = ctx->saved_data["dt"].toDouble();
        bool is_log_space = ctx->saved_data["is_log_space"].toBool();
        
        // Ensure gradient is contiguous
        auto grad_output = grad_outputs[0].contiguous();
        
        // Debug output for gradient
        // std::cout << "RK4 grad_output norm: " 
        //           << grad_output.norm().item<float>() << std::endl;
        
        // Check for NaN, Inf, or extreme values
        bool has_nan = grad_output.isnan().any().item<bool>();
        bool has_inf = grad_output.isinf().any().item<bool>();
        float grad_norm = grad_output.norm().item<float>();
        bool extreme_values = grad_norm > 1000.0;
        
        if (has_nan || has_inf || extreme_values) {
            // std::cout << "WARNING: RK4 gradient contains ";
            // if (has_nan) std::cout << "NaN";
            // else if (has_inf) std::cout << "Inf";
            // else std::cout << "extreme values (" << grad_norm << ")";
            // std::cout << "! Fixing..." << std::endl;
            
            // Replace problematic values
            grad_output = torch::nan_to_num(grad_output, 0.0, 1.0, -1.0);
            
            // Clip extreme values
            grad_output = clip_gradient_norm(grad_output, 100.0);
        }
        
        // Call the backward implementation
        auto grads = rk4_step_backward_cuda(
            grad_output, population, k1, k2, k3, k4, dt, is_log_space
        );
        
        // Debug output and fix any issues with the gradients
        for (size_t i = 0; i < grads.size(); ++i) {
            if (grads[i].defined()) {
                // Check for NaN/Inf
                if (grads[i].isnan().any().item<bool>() || grads[i].isinf().any().item<bool>()) {
                    std::cout << "WARNING: NaN/Inf in RK4 grad[" << i << "]! Fixing..." << std::endl;
                    grads[i] = torch::nan_to_num(grads[i], 0.0, 1.0, -1.0);
                }
                
                // Clip gradient norm
                grads[i] = clip_gradient_norm(grads[i], 1000.0);
                
                // std::cout << "RK4 grad[" << i << "] norm: " 
                //          << grads[i].norm().item<float>() << std::endl;
            }
        }
        
        // Return gradients for each input (non-tensor inputs get None)
        return {grads[0], grads[1], grads[2], grads[3], grads[4], torch::Tensor(), torch::Tensor()};
    }
};

//========================================================================
// Python module binding
//========================================================================

// Forward function wrappers that use autograd

torch::Tensor gradient_cuda(
    torch::Tensor population,
    torch::Tensor da_coordinates,
    int max_neighbors = 8
) {
    return GradientFunction::apply(population, da_coordinates, max_neighbors);
}

torch::Tensor divergence_cuda(
    torch::Tensor vector_field,
    torch::Tensor da_coordinates,
    int max_neighbors = 4
) {
    return DivergenceFunction::apply(vector_field, da_coordinates, max_neighbors);
}

std::vector<torch::Tensor> pde_terms_cuda(
    torch::Tensor population,
    torch::Tensor gradient,
    torch::Tensor diffusion,
    torch::Tensor amplification,
    torch::Tensor velocity,
    torch::Tensor alpha,
    torch::Tensor beta,
    torch::Tensor gamma,
    torch::Tensor laplacian,
    torch::Tensor velocity_div,
    double sigma_squared,
    double K,
    double pop_scale_factor
) {
    return PDETermsFunction::apply(
        population, gradient, diffusion, amplification, velocity,
        alpha, beta, gamma, laplacian, velocity_div,
        sigma_squared, K, pop_scale_factor
    );
}

torch::Tensor rk4_step_cuda_autograd(
    torch::Tensor population,
    torch::Tensor k1,
    torch::Tensor k2,
    torch::Tensor k3,
    torch::Tensor k4,
    double dt,
    bool is_log_space = false
) {
    return RK4StepFunction::apply(population, k1, k2, k3, k4, dt, is_log_space);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Original (backward compatible) functions
    m.def("compute_gradient", &compute_gradient_cuda, "Compute gradient (CUDA)",
          py::arg("population"), py::arg("da_coordinates"), py::arg("max_neighbors") = 8);
    m.def("compute_divergence", &compute_divergence_cuda, "Compute divergence (CUDA)",
          py::arg("vector_field"), py::arg("da_coordinates"), py::arg("max_neighbors") = 4);
    m.def("compute_pde_terms", &compute_pde_terms_cuda, "Compute PDE terms (CUDA)",
          py::arg("population"), py::arg("gradient"), py::arg("diffusion"), 
          py::arg("amplification"), py::arg("velocity"), py::arg("alpha"), 
          py::arg("beta"), py::arg("gamma"), py::arg("laplacian"), 
          py::arg("velocity_div"), py::arg("sigma_squared"), 
          py::arg("K"), py::arg("pop_scale_factor"));
    m.def("rk4_step", &rk4_step_cuda, "RK4 integration step (CUDA)",
          py::arg("population"), py::arg("k1"), py::arg("k2"), 
          py::arg("k3"), py::arg("k4"), py::arg("dt"));
    
    // New autograd-enabled functions
    m.def("gradient", &gradient_cuda, "Compute gradient with autograd (CUDA)",
          py::arg("population"), py::arg("da_coordinates"), py::arg("max_neighbors") = 8);
    m.def("divergence", &divergence_cuda, "Compute divergence with autograd (CUDA)",
          py::arg("vector_field"), py::arg("da_coordinates"), py::arg("max_neighbors") = 4);
    m.def("pde_terms", &pde_terms_cuda, "Compute PDE terms with autograd (CUDA)",
          py::arg("population"), py::arg("gradient"), py::arg("diffusion"), 
          py::arg("amplification"), py::arg("velocity"), py::arg("alpha"), 
          py::arg("beta"), py::arg("gamma"), py::arg("laplacian"), 
          py::arg("velocity_div"), py::arg("sigma_squared"), 
          py::arg("K"), py::arg("pop_scale_factor"));
    m.def("rk4_step_autograd", &rk4_step_cuda_autograd, "RK4 integration step with autograd (CUDA)",
          py::arg("population"), py::arg("k1"), py::arg("k2"), 
          py::arg("k3"), py::arg("k4"), py::arg("dt"),
          py::arg("is_log_space") = false);
}