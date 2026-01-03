import torch
import torch.nn as nn

from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
    point_mesh_face_distance

)

class ShapeCompletionLoss(nn.Module):
    def __init__(self, sample_size=2048, w_mse=1.0, w_chamfer=0.33, w_edge=0.1, w_normal=0.01, w_laplacian=0.1, w_point_mesh_dist=0.1, seed=42):
        super().__init__()
        self.sample_size = sample_size
        self.w_chamfer = w_chamfer
        self.w_edge = w_edge
        self.w_normal = w_normal
        self.w_laplacian = w_laplacian
        self.w_point_mesh_dist = w_point_mesh_dist
        self.seed = seed

    def forward(self, pred_meshes, gt_meshes, eps_pred, noise, deterministic=False):
        """
        Args:
            pred_meshes: pytorch3d.structures.Meshes object
            gt_meshes: pytorch3d.structures.Meshes object
            deterministic: Sample points deterministically (used during inference)
        """

        if deterministic:
            # fork_rng ensures we don't mess up global randomness for the rest of training
            with torch.random.fork_rng():
                torch.manual_seed(self.seed)
                pred_sampled, pred_normals = sample_points_from_meshes(
                    pred_meshes, num_samples=self.sample_size, return_normals=True
                )

                torch.manual_seed(self.seed)
                gt_sampled, gt_normals = sample_points_from_meshes(
                    gt_meshes, num_samples=self.sample_size, return_normals=True
                )
        else:
            # Standard stochastic sampling for training (Recommended)
            pred_sampled, pred_normals = sample_points_from_meshes(
                pred_meshes, num_samples=self.sample_size, return_normals=True
            )
            gt_sampled, gt_normals = sample_points_from_meshes(
                gt_meshes, num_samples=self.sample_size, return_normals=True
            )

        # disregard normal loss
        loss_chamfer, _ = chamfer_distance(
            pred_sampled,
            gt_sampled,
            x_normals=pred_normals,
            y_normals=gt_normals
        )

        # Also add a loss such that the mesh is close to the gt point cloud
        loss_point_mesh_dist = point_mesh_face_distance(meshes=pred_meshes, pcls=gt_sampled)

        # --- Regularization Losses --- #

        # Edge Loss: Penalizes long edges to prevent "flying vertices"
        loss_edge = mesh_edge_loss(pred_meshes)

        # Normal Consistency: Penalizes sharp changes in face normals (smoothness)
        loss_normal = mesh_normal_consistency(pred_meshes)

        # Laplacian Smoothing: Penalizes irregular vertex placement (smoothness)
        loss_laplacian = mesh_laplacian_smoothing(pred_meshes, method="uniform")

        mse_loss = nn.MSELoss(eps_pred, noise)
        total_loss = (
                self.w_chamfer * loss_chamfer +
                self.w_edge * loss_edge +
                self.w_normal * loss_normal +
                self.w_laplacian * loss_laplacian +
                self.w_point_mesh_dist * loss_point_mesh_dist +
                self.w_mse * mse_loss
        )

        return total_loss, {
            "chamfer": loss_chamfer,
            "edge": loss_edge,
            "normal": loss_normal,
            "laplacian": loss_laplacian,
            "point_mesh_dist": loss_point_mesh_dist,
            "mse": mse_loss,
        }