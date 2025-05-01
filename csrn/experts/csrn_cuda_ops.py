import torch
import warnings

# Try to import the custom CUDA extension
try:
    import csrn_cuda
    CUDA_AVAILABLE = True
    print("CSRN CUDA extension loaded successfully with autograd support.")
except ImportError:
    CUDA_AVAILABLE = False
    warnings.warn(
        "CSRN CUDA extension not available. Using PyTorch fallback implementation. "
        "This will be slower but should work for development."
    )

class CSRNCudaFallback:
    """
    Fallback implementation of CUDA operations using standard PyTorch.
    This implementation is slower but works without the CUDA extension.
    It supports autograd by using PyTorch operations.
    """
    
    @staticmethod
    def compute_gradient(population, da_coordinates, max_neighbors=8):
        """
        PyTorch fallback for gradient computation with autograd support.
        """
        # Ensure tensors are on the correct device and type
        if not isinstance(population, torch.Tensor):
            population = torch.tensor(population, dtype=torch.float32, device='cuda')
        else:
            population = population.to(torch.float32).cuda()
            
        if not isinstance(da_coordinates, torch.Tensor):
            da_coordinates = torch.tensor(da_coordinates, dtype=torch.float32).cuda()
        else:
            da_coordinates = da_coordinates.to(torch.float32).cuda()
        
        batch_size = population.shape[0]
        num_das = population.shape[1]
        
        # Get min/max coordinates for normalization
        coords_min, _ = torch.min(da_coordinates, dim=0)
        coords_max, _ = torch.max(da_coordinates, dim=0)
        coords_extent = coords_max - coords_min
        
        # Normalize coordinates
        norm_coords = (da_coordinates - coords_min) / coords_extent
        
        # Initialize gradient tensor
        gradient = torch.zeros((batch_size, num_das, 2), dtype=torch.float32, device=population.device)
        
        # Loop through each DA
        for b in range(batch_size):
            for i in range(num_das):
                # Compute distances to all other DAs
                dists = torch.sum((norm_coords - norm_coords[i].unsqueeze(0))**2, dim=1)
                
                # Sort distances and get indices of nearest neighbors
                sorted_dists, indices = torch.sort(dists)
                
                # Skip self (index 0) and limit to max_neighbors
                indices = indices[1:min(max_neighbors+1, len(indices))]
                
                if len(indices) < 2:  # Need at least 2 neighbors
                    continue
                
                # Compute gradient
                grad_sum = torch.zeros(2, dtype=torch.float32, device=population.device)
                weight_sum = 0.0
                
                for j in indices:
                    # Direction vector
                    dir_vec = norm_coords[j] - norm_coords[i]
                    dist_squared = torch.sum(dir_vec**2)
                    
                    # Skip very small distances
                    if dist_squared < 1e-8:
                        continue
                    
                    dist = torch.sqrt(dist_squared)
                    dir_vec = dir_vec / dist
                    
                    # Population difference
                    pop_diff = population[b, j] - population[b, i]
                    
                    # Weight by inverse squared distance
                    weight = torch.clamp(1.0 / dist_squared, max=100.0)
                    
                    # Contribution to gradient
                    contribution = weight * pop_diff * dir_vec
                    
                    # Add if valid
                    if torch.isfinite(contribution).all():
                        grad_sum += contribution
                        weight_sum += weight
                
                # Normalize by total weight
                if weight_sum > 1e-10:
                    gradient[b, i] = torch.clamp(grad_sum / weight_sum, -1e4, 1e4)
        
        return gradient
    
    @staticmethod
    def compute_divergence(vector_field, da_coordinates, max_neighbors=4):
        """
        PyTorch fallback for divergence computation with autograd support.
        """
        if not isinstance(vector_field, torch.Tensor):
            vector_field = torch.tensor(vector_field, dtype=torch.float32, device='cuda')
        else:
            vector_field = vector_field.to(torch.float32).cuda()
            
        if not isinstance(da_coordinates, torch.Tensor):
            da_coordinates = torch.tensor(da_coordinates, dtype=torch.float32).cuda()
        else:
            da_coordinates = da_coordinates.to(torch.float32).cuda()
        
        batch_size = vector_field.shape[0]
        num_das = vector_field.shape[1]
        
        # Get min/max coordinates for normalization
        coords_min, _ = torch.min(da_coordinates, dim=0)
        coords_max, _ = torch.max(da_coordinates, dim=0)
        coords_extent = coords_max - coords_min
        
        # Normalize coordinates
        norm_coords = (da_coordinates - coords_min) / coords_extent
        
        # Initialize divergence tensor
        divergence = torch.zeros((batch_size, num_das), dtype=torch.float32, device=vector_field.device)
        
        # Loop through each DA
        for b in range(batch_size):
            for i in range(num_das):
                # Compute distances to all other DAs
                dists = torch.sum((norm_coords - norm_coords[i].unsqueeze(0))**2, dim=1)
                
                # Sort distances and get indices of nearest neighbors
                sorted_dists, indices = torch.sort(dists)
                
                # Skip self (index 0) and limit to max_neighbors
                indices = indices[1:min(max_neighbors+1, len(indices))]
                
                if len(indices) < 2:  # Need at least 2 neighbors
                    continue
                
                # Compute divergence
                div_sum = 0.0
                neighbor_count = 0
                
                for j in indices:
                    # Direction vector
                    dir_vec = norm_coords[j] - norm_coords[i]
                    dist_squared = torch.sum(dir_vec**2)
                    
                    # Skip very small distances
                    if dist_squared < 1e-6:
                        continue
                    
                    dist = torch.sqrt(dist_squared)
                    dir_vec = dir_vec / dist
                    
                    # Vector field difference
                    field_diff = vector_field[b, j] - vector_field[b, i]
                    
                    # Directional derivative
                    div_contrib = torch.dot(field_diff, dir_vec) / dist
                    
                    # Add if valid
                    if torch.isfinite(div_contrib):
                        div_sum += div_contrib
                        neighbor_count += 1
                
                # Normalize by number of neighbors
                if neighbor_count > 0:
                    divergence[b, i] = torch.clamp(div_sum / neighbor_count, -10.0, 10.0)
        
        return divergence
    
    @staticmethod
    def compute_pde_terms(
        population,
        gradient,
        diffusion,
        amplification,
        velocity,
        alpha,
        beta,
        gamma,
        laplacian,
        velocity_div,
        sigma_squared,
        K,
        pop_scale_factor
    ):
        """
        PyTorch fallback for PDE terms computation with autograd support.
        """
        print("WARNING!!!!!!!! USING FALLBACK")
        # Ensure tensors are on GPU
        tensors = [population, gradient, diffusion, amplification, velocity,
                  alpha, beta, gamma, laplacian, velocity_div]
        tensors = [t.to('cuda') if isinstance(t, torch.Tensor) else torch.tensor(t, device='cuda')
                  for t in tensors]
        
        population, gradient, diffusion, amplification, velocity, alpha, beta, gamma, laplacian, velocity_div = tensors
        
        # Compute gradient magnitude squared
        grad_mag_squared = torch.sum(gradient**2, dim=-1)
        grad_mag_squared = torch.clamp(grad_mag_squared, 0.0, 100.0)
        
        # Diffusion term
        diffusion_term = diffusion * laplacian * pop_scale_factor
        
        # Directed flow term
        # Use batch matrix operations to avoid loops
        batch_size, num_das = population.shape
        directed_flow = torch.zeros_like(population)
        
        for b in range(batch_size):
            # Dot product of velocity and gradient for each DA
            directed_flow[b] = amplification[b] * torch.sum(velocity[b] * gradient[b], dim=1)
        
        # Apply spatial dependency for alpha
        exp_term = torch.exp(torch.clamp(-grad_mag_squared / sigma_squared, min=-50.0, max=50.0))
        alpha_term = alpha * exp_term
        
        # Compute mean reversion term
        pop_mean = torch.mean(population, dim=1, keepdim=True)
        mean_reversion = -alpha_term * (population - pop_mean)
        
        # Compute gradient control term
        gradient_penalty = -beta * torch.clamp(grad_mag_squared - K, min=0.0)
        
        # Compute advection stabilization term
        advection_stabilization = -gamma * velocity_div * population * pop_scale_factor * 0.1
        
        return [diffusion_term, directed_flow, mean_reversion, gradient_penalty, advection_stabilization]
    
    @staticmethod
    def rk4_step(population, k1, k2, k3, k4, dt, is_log_space=False):
        """
        PyTorch fallback for RK4 integration step with autograd support.
        """
        # Ensure tensors are on GPU
        tensors = [population, k1, k2, k3, k4]
        tensors = [t.to('cuda') if isinstance(t, torch.Tensor) else torch.tensor(t, device='cuda')
                  for t in tensors]
        
        population, k1, k2, k3, k4 = tensors
        
        # RK4 update
        rk4_sum = k1 + 2.0 * k2 + 2.0 * k3 + k4
        updated_pop = population + (dt / 6.0) * rk4_sum
        
        # In log space we allow negative values, otherwise enforce non-negative
        if not is_log_space:
            updated_pop = torch.clamp(updated_pop, min=0.0)
        
        return updated_pop

class CSRNCudaOps:
    """
    CUDA operations for the Cultural-Spatial Resonance Network with autograd support.
    This class provides accelerated versions of the most computationally expensive
    operations in the PDE solver that are differentiable.
    """
    
    @staticmethod
    def compute_gradient(population, da_coordinates, max_neighbors=8):
        """
        CUDA-accelerated gradient computation with autograd.
        """
        # Make sure the input tensors require gradients
        requires_grad = getattr(population, 'requires_grad', False)
        
        if CUDA_AVAILABLE:
            # Use the autograd-enabled version
            result = csrn_cuda.gradient(population.contiguous(), da_coordinates, max_neighbors)
            
            # Verify the result has requires_grad set properly
            if requires_grad and not result.requires_grad:
                warnings.warn("Gradient result doesn't have requires_grad=True despite input having it")
            
            return result
        else:
            return CSRNCudaFallback.compute_gradient(population, da_coordinates, max_neighbors)
    
    @staticmethod
    def compute_divergence(vector_field, da_coordinates, max_neighbors=4):
        """
        CUDA-accelerated divergence computation with autograd.
        """
        if CUDA_AVAILABLE:
            # Use the autograd-enabled version
            return csrn_cuda.divergence(vector_field.contiguous(), da_coordinates, max_neighbors)
        else:
            return CSRNCudaFallback.compute_divergence(vector_field, da_coordinates, max_neighbors)
    
    @staticmethod
    def compute_pde_terms(
        population,
        gradient,
        diffusion,
        amplification,
        velocity,
        alpha,
        beta,
        gamma,
        laplacian,
        velocity_div,
        sigma_squared,
        K,
        pop_scale_factor
    ):
        """
        CUDA-accelerated PDE terms computation with autograd.
        """
        if CUDA_AVAILABLE:
            # Use the autograd-enabled version
            return csrn_cuda.pde_terms(
                population.contiguous(),
                gradient,
                diffusion.contiguous(),
                amplification.contiguous(),
                velocity.contiguous(),
                alpha.contiguous(),
                beta.contiguous(),
                gamma.contiguous(),
                laplacian.contiguous(),
                velocity_div.contiguous(),
                float(sigma_squared),
                float(K),
                float(pop_scale_factor)
            )
        else:
            return CSRNCudaFallback.compute_pde_terms(
                population,
                gradient,
                diffusion,
                amplification,
                velocity,
                alpha,
                beta,
                gamma,
                laplacian,
                velocity_div,
                sigma_squared,
                K,
                pop_scale_factor
            )
    
    @staticmethod
    def rk4_step(population, k1, k2, k3, k4, dt, is_log_space=False):
        """
        CUDA-accelerated RK4 integration step with autograd.
        """
        if CUDA_AVAILABLE:
            # Use the autograd-enabled version with is_log_space flag
            return csrn_cuda.rk4_step_autograd(
                population,
                k1,
                k2,
                k3,
                k4,
                float(dt),
                is_log_space
            )
        else:
            return CSRNCudaFallback.rk4_step(population, k1, k2, k3, k4, dt, is_log_space)
    
    @staticmethod
    def test_gradient_flow():
        """
        Test function to verify gradient flow through the CUDA operations.
        """
        try:
            # Create random test data
            batch_size = 2
            num_das = 10
            
            # Create tensors with requires_grad=True
            population = torch.rand(batch_size, num_das, requires_grad=True, device='cuda')
            coords = torch.rand(num_das, 2, device='cuda')
            
            # Compute gradient
            gradient = CSRNCudaOps.compute_gradient(population, coords)
            
            # Ensure the gradient is contiguous before backward pass
            gradient = gradient.contiguous()
            
            # Verify gradient tensor has requires_grad
            if not gradient.requires_grad:
                warnings.warn("Gradient tensor doesn't have requires_grad=True")
                return False
            
            # Test backward pass with explicit error handling
            loss = gradient.sum()
            try:
                loss.backward()
            except Exception as e:
                warnings.warn(f"Backward pass failed: {str(e)}")
                return False
            
            # Verify population has gradients
            if population.grad is None:
                warnings.warn("Population tensor doesn't have gradients")
                return False
            else:
                print("Gradient flow test passed: Population has gradients")
                return True
        
        except Exception as e:
            warnings.warn(f"Gradient flow test failed with error: {str(e)}")
            return False

# Run a test when the module is imported
if CUDA_AVAILABLE:
    print("Testing gradient flow through custom CUDA operations...")
    CSRNCudaOps.test_gradient_flow()