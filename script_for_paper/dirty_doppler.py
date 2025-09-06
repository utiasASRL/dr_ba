
import cv2
import numpy as np
from sklearn.metrics import pairwise_distances
import torch
import torchvision

import pyboreas as pb


class LocalMapRegistrator:
    def __init__(self, source, target, res, xytheta_init=np.array([0, 0, 0])):

        # Check the input shapes match and that the nb of collumn and rows are odd
        if source.shape[0] != target.shape[0] or source.shape[1] != target.shape[1] or source.shape[0] % 2 == 0 or source.shape[1] % 2 == 0:
            raise ValueError("Source and target images must have the same shape and odd dimensions")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.optimisation_first_step = 0.1

        with torch.no_grad():
            self.source = torch.tensor(source, device=self.device).float()
            self.target = torch.tensor(target, device=self.device).float()
            self.res = torch.tensor(res, device=self.device).float()
            self.xytheta_init = torch.tensor(xytheta_init, device=self.device).float()

            # Create the cartesian coordinates that correspond to each pixel of the images
            self.cartesian_coords = torch.zeros((self.source.shape[0], self.source.shape[1], 2, 1), device=self.device)
            self.cartesian_coords[:, :, 0, 0] = -((torch.arange(self.source.shape[0], device=self.device)- (self.source.shape[0] // 2)).float() * self.res).reshape((-1, 1))
            self.cartesian_coords[:, :, 1, 0] = ((torch.arange(self.source.shape[1], device=self.device)- (self.source.shape[1] // 2)).float() * self.res).reshape((1, -1))


    def cartToImageID_(self, xy):
        with torch.no_grad():
            out = torch.empty_like(xy, device=self.device)
            out[:,:,0,0] = (xy[:,:,0,0] / (-self.res)) + (self.source.shape[0] // 2)
            out[:,:,1,0] = (xy[:,:,1,0] / (self.res)) + (self.source.shape[1] // 2)
            gradient = torch.tensor([[-1.0/self.res, 0], [0, 1.0/self.res]], device=self.device).reshape((1,1,2,2))
            return out, gradient


    def transformSource_(self, xytheta):
        with torch.no_grad():
            c_rot = torch.cos(xytheta[2])
            s_rot = torch.sin(xytheta[2])
            rot_mat_T = torch.tensor([[c_rot, -s_rot], [s_rot, c_rot]], device=self.device).T.reshape((1,1, 2, 2))
            pos = xytheta[:2].reshape((1,1, 2, 1)).to(self.device)

            # Transform the cartesian coordinates
            cartesian_coords_transformed = rot_mat_T @ self.cartesian_coords - rot_mat_T @ pos

            # Convert the cartesian coordinates to image coordinates
            ids, gradient = self.cartToImageID_(cartesian_coords_transformed)

            # Get the interpolated source image
            source_interp = self.bilinearInterpolation_(self.source, ids.squeeze(), with_jac=False)

            # Residuals
            residuals = source_interp * self.source

            return source_interp, residuals


    def bilinearInterpolation_(self, im, az_r, with_jac = False):
        with torch.no_grad():
            az0 = torch.floor(az_r[:, :, 0]).int()
            az1 = az0 + 1
            
            r0 = torch.floor(az_r[:, :, 1]).int()
            r1 = r0 + 1

            az0 = torch.clamp(az0, 0, im.shape[0]-1)
            az1 = torch.clamp(az1, 0, im.shape[0]-1)
            r0 = torch.clamp(r0, 0, im.shape[1]-1)
            r1 = torch.clamp(r1, 0, im.shape[1]-1)
            az_r[:,:,0] = torch.clamp(az_r[:,:,0], 0, im.shape[0]-1)
            az_r[:,:,1] = torch.clamp(az_r[:,:,1], 0, im.shape[1]-1)
            
            Ia = im[ az0, r0 ]
            Ib = im[ az1, r0 ]
            Ic = im[ az0, r1 ]
            Id = im[ az1, r1 ]
            
            local_1_minus_r = (r1.float()-az_r[:, :, 1])
            local_r = (az_r[:, :, 1]-r0.float())
            local_1_minus_az = (az1.float()-az_r[:, :, 0])
            local_az = (az_r[:, :, 0]-az0.float())
            wa = local_1_minus_az * local_1_minus_r
            wb = local_az * local_1_minus_r
            wc = local_1_minus_az * local_r
            wd = local_az * local_r

            img_interp = wa*Ia + wb*Ib + wc*Ic + wd*Id

            if not with_jac:
                return img_interp
            else:
                d_I_d_az_r = torch.empty((az_r.shape[0], az_r.shape[1], 1, 2), device=self.device)
                d_I_d_az_r[:, :, 0, 0] = (Ib - Ia)*local_1_minus_r + (Id - Ic)*local_r
                d_I_d_az_r[:, :, 0, 1] = (Ic - Ia)*local_1_minus_az + (Id - Ib)*local_az
                return img_interp, d_I_d_az_r
        


    def register(self, nb_iter=20, cost_tol=1e-6, step_tol=1e-6, verbose=False, degraded=False):
        with torch.no_grad():
            # The gradient ascent keep track of the last increasing state and gradient
            # Thus, if the cost function decreases, we go back to the last increasing
            # state and reduce the step size
            state = self.xytheta_init.clone().to(self.device).float()
            first_cost = torch.tensor(np.inf).to(self.device)
            prev_cost = first_cost
            first_quantum = self.optimisation_first_step
            step_quantum = first_quantum
            last_increasing_state = state.clone()
            last_increasing_grad = torch.zeros_like(state)
            for i in torch.arange(nb_iter, device=self.device):
                
                res, jac = self.costFunctionAndJacobian(state)

                #grad = 3*torch.sum(res.flatten().unsqueeze(-1)**2 * jac.reshape((-1,jac.shape[-1])), 0)
                #cost = torch.sum((res**3).flatten())
                grad = torch.sum(jac, 0)
                cost = torch.sum((res).flatten())

                if i == 0:
                    last_increasing_grad = grad.clone()
                else:
                    if cost < prev_cost:
                        state = last_increasing_state.clone()
                        grad = last_increasing_grad.clone()
                        step_quantum = step_quantum / 2
                    else:
                        last_increasing_state = state.clone()
                        last_increasing_grad = grad.clone()

                grad_norm = torch.linalg.norm(grad)

                if step_quantum < 1e-5:
                    break


                if grad_norm < 1e-9:
                    break
                step = (grad / grad_norm) * step_quantum
                
                state += step

                step_norm = torch.linalg.norm(step)
                cost_change = cost - prev_cost

                if i == 0:
                    first_cost = cost
                
                # Print iter cost step_norm cost_change with 3 decimals and scientific notation
                if verbose:
                    print("Iter: ", i, " - Cost: ", "{:.3e}".format(cost), " - Step norm: ", "{:.3e}".format(step_norm), " - Cost change: ", "{:.3e}".format(cost_change))

                if step_norm < step_tol:
                    break

                if torch.abs(cost_change/cost) < cost_tol:
                    break
                prev_cost = cost

            state_np = state.detach().cpu().numpy()

            self.xytheta_init = state.clone()

            return state_np


    def costFunctionAndJacobian(self, xytheta, with_jac=True):
        with torch.no_grad():
            # Get the rotation matrix
            c_rot = torch.cos(xytheta[2])
            s_rot = torch.sin(xytheta[2])
            rot_mat = torch.tensor([[c_rot, -s_rot], [s_rot, c_rot]], device=self.device).reshape((1,1, 2, 2))
            pos = xytheta[:2].reshape((1,1, 2, 1)).to(self.device)

            # Transform the cartesian coordinates
            cartesian_coords_transformed = rot_mat @ self.cartesian_coords

            if with_jac:
                d_cartesian_coords_transformed_d_state = torch.zeros((self.cartesian_coords.shape[0], self.cartesian_coords.shape[1], 2, 3), device=self.device)
                d_cartesian_coords_transformed_d_state[:,:,0, 0] = 1
                d_cartesian_coords_transformed_d_state[:,:,1, 1] = 1
                d_cartesian_coords_transformed_d_state[:,:,0, 2] = -cartesian_coords_transformed[:,:,1,0]
                d_cartesian_coords_transformed_d_state[:,:,1, 2] = cartesian_coords_transformed[:,:,0,0]
            cartesian_coords_transformed += pos

            # Convert the cartesian coordinates to image coordinates
            ids, gradient = self.cartToImageID_(cartesian_coords_transformed)
            if with_jac:
                d_ids_dstate = gradient @ d_cartesian_coords_transformed_d_state


            # Get the interpolated source image
            if with_jac:
                source_interp, d_source_interp = self.bilinearInterpolation_(self.target, ids.squeeze(), with_jac=True)
                d_source_interp = d_source_interp @ d_ids_dstate
            else:
                source_interp = self.bilinearInterpolation_(self.target, ids.squeeze(), with_jac=False)

            # Residuals
            residuals = source_interp * self.source
            if with_jac:
                gradient = self.source.unsqueeze(-1).unsqueeze(-1) @ d_source_interp
                return residuals.flatten(), gradient.reshape((-1,3))
            else:
                return residuals.flatten()


    def testCostFunctionGrad(self):
        state = torch.rand(3, device=self.device)
        cost, grad = self.costFunctionAndJacobian(state)
        cost_np = cost.cpu().numpy()
        grad_np = grad.cpu().numpy()

        epsilon = 1e-3
        grad_fd = np.zeros_like(grad_np)
        for i in range(3):
            state_p = state.clone()
            state_p[i] += epsilon
            cost_p, _ = self.costFunctionAndJacobian(state_p)
            cost_p = cost_p.cpu().numpy()

            state_m = state.clone()
            state_m[i] -= epsilon
            cost_m, _ = self.costFunctionAndJacobian(state_m)
            cost_m = cost_m.cpu().numpy()

            grad_fd[:, i] = (cost_p - cost_m) / (2 * epsilon)
        
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(3, 2)
        for i in range(3):
            axs[i,0].plot(grad_np[:, i], label='Analytical', linewidth=0.5)
            axs[i,0].plot(grad_fd[:, i], label='Finite Difference', linewidth=0.5)
            axs[i,0].legend()
            axs[i,0].set_title(f'Gradient component {i}')
            axs[i,1].plot(grad_np[:, i] - grad_fd[:, i], label='Difference', linewidth=0.5)
            axs[i,1].legend()
            axs[i,1].set_title(f'Gradient difference {i}')
        plt.show()


    def getOverlay(self):
        # Display the overlay of the source and target images
        source_interp, _ = self.transformSource_(self.xytheta_init)

        return source_interp.detach().cpu().numpy()



    def getRegistrationScore(self):
        # Compute the registration
        with torch.no_grad():
            residuals = self.costFunctionAndJacobian(self.xytheta_init, with_jac=False)
            return torch.sum(residuals) / torch.sum(self.target**2)
    
    def gridSearchInitialization(self, search_ranges, nb_steps):
        with torch.no_grad():
            xs = torch.linspace(search_ranges[0][0], search_ranges[0][1], nb_steps, device=self.device) + self.xytheta_init[0]
            ys = torch.linspace(search_ranges[1][0], search_ranges[1][1], nb_steps, device=self.device) + self.xytheta_init[1]
            thetas = torch.linspace(search_ranges[2][0], search_ranges[2][1], nb_steps, device=self.device) + self.xytheta_init[2]
            best_cost = -np.inf
            best_state = self.xytheta_init.clone()
            for x in xs:
                for y in ys:
                    for theta in thetas:
                        cost = self.costFunctionAndJacobian(torch.tensor([x, y, theta], device=self.device), with_jac=False)
                        cost = torch.sum(cost)
                        if cost > best_cost:
                            best_cost = cost
                            best_state = torch.tensor([x, y, theta], device=self.device)
            self.xytheta_init = best_state