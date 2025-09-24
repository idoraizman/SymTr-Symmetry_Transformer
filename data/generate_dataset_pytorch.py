import torch
import cv2
import numpy as np
from tqdm import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import argparse

def _random_line(batch_size=1, device='cpu'):
    """Generate random lines in normal form (cosθ, sinθ, d)."""
    theta = torch.rand(batch_size, device=device) * torch.pi
    n = torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)  # (batch_size, 2)
    c = torch.rand(batch_size, device=device) * 2.0 - 1.0  # d in [-1, 1]
    d = torch.stack([-n[:, 1], n[:, 0]], dim=1)  # direction vector (batch_size, 2)
    return n, c, d

def _generate_polynomial_shape(n, c, d, num_curve_pts=5000, min_deg=3, max_deg=5, device='cpu', visualize=False):
    """Generate symmetric curves around a line using a random polynomial.
    
    Args:
        n: Normal vector to the line (batch_size, 2).
        c: Center point of the line (batch_size, 2).
        d: Direction vector of the line (batch_size, 2).
        num_curve_pts: Number of points to sample along the curve.
        min_deg: Minimum polynomial degree.
        max_deg: Maximum polynomial degree.
        device: Device for computation ('cpu' or 'cuda').
        visualize: If True, display the line, curve, and reflected curve for each batch.
    
    Returns:
        curve: Tensor of shape (batch_size, num_curve_pts, 2) for the curve.
        reflected: Tensor of shape (batch_size, num_curve_pts, 2) for the reflected curve.
    """
    # Ensure inputs are tensors and validate shapes
    n = torch.as_tensor(n, device=device, dtype=torch.float32)
    c = torch.as_tensor(c, device=device, dtype=torch.float32)
    d = torch.as_tensor(d, device=device, dtype=torch.float32)
    
    # Check shapes
    batch_size = n.shape[0]
    if n.shape != (batch_size, 2):
        raise ValueError(f"Expected n to have shape (batch_size, 2), got {n.shape}")
    if d.shape != (batch_size, 2):
        raise ValueError(f"Expected d to have shape (batch_size, 2), got {d.shape}")
    
    # Random intersection points t1, t2
    t1 = torch.rand(batch_size, device=device) * 0.5 - 1.0  # [-1, -0.5]
    t2 = torch.rand(batch_size, device=device) * 0.5 + 0.5  # [0.5, 1]
    
    # Random polynomial degree
    deg = torch.randint(min_deg, max_deg, (batch_size,), device=device)
    max_deg = max_deg - 1
    
    # Generate roots for polynomial
    roots = torch.rand(batch_size, max_deg, device=device)
    roots = roots * (t2 - t1).unsqueeze(1) * 0.8 + t1.unsqueeze(1) + 0.1 * (t2 - t1).unsqueeze(1)
    
    # Polynomial amplitude and offset
    amp = torch.rand(batch_size, device=device) * 6 * deg.float() + 6 * deg.float()
    offset = torch.rand(batch_size, device=device) * 0.4 - 0.2  # [-0.2, 0.2]
    
    # Sample points along the line
    u = torch.linspace(-1, 1, num_curve_pts, device=device).unsqueeze(0).expand(batch_size, num_curve_pts)
    u = u * (t2 - t1).unsqueeze(1) / 2 + (t1 + t2).unsqueeze(1) / 2
    
    # Evaluate polynomial
    v = torch.ones_like(u)
    for i in range(max_deg):
        mask = (i < deg.unsqueeze(1)).float()
        v = v * (u - roots[:, i].unsqueeze(1)) * mask + (1 - mask) * v
    v = v + offset.unsqueeze(1)
    v = v * (u - t1.unsqueeze(1)) * (u - t2.unsqueeze(1)) * amp.unsqueeze(1)
    
    # Generate points
    base = (c.unsqueeze(1)*n).unsqueeze(1) + u.unsqueeze(2) * d.unsqueeze(1)
    curve = base + v.unsqueeze(2) * n.unsqueeze(1)
    reflected = base - v.unsqueeze(2) * n.unsqueeze(1)
    
    if visualize:
        # Convert tensors to numpy for plotting
        curve_np = curve.cpu().numpy()
        reflected_np = reflected.cpu().numpy()
        c_np = c.cpu().numpy()
        n_np = n.cpu().numpy()
        d_np = d.cpu().numpy()
        t1_np = t1.cpu().numpy()
        t2_np = t2.cpu().numpy()
        
        # Create a subplot for each batch element
        cols = min(batch_size, 3)  # Max 3 columns for readability
        rows = (batch_size + cols - 1) // cols
        plt.figure(figsize=(5 * cols, 5 * rows))
        
        for i in range(batch_size):
            plt.subplot(rows, cols, i + 1)
            
            # Plot the base line from t1 to t2
            t = np.linspace(t1_np[i], t2_np[i], 100)
            line_pts = c_np[i]*n_np[i] + t[:, np.newaxis] * d_np[i]
            plt.plot(line_pts[:, 0], line_pts[:, 1], 'k-', label='Base Line', alpha=0.5)
            
            # Plot the curve
            plt.plot(curve_np[i, :, 0], curve_np[i, :, 1], 'b-', label='Curve')
            
            # Plot the reflected curve
            plt.plot(reflected_np[i, :, 0], reflected_np[i, :, 1], 'r-', label='Reflected Curve')
            
            # Plot center point
            # plt.scatter(c_np[i, 0], c_np[i, 1], color='green', label='Center', marker='o')
            
            plt.title(f'Batch {i + 1}')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.legend()
            plt.grid(True)
            plt.axis('equal')  # Equal aspect ratio for better visualization
        
        plt.tight_layout()
        plt.show()
    
    return curve, reflected  # (batch_size, num_curve_pts, 2)


def point_cloud_to_outline(points, img_size=512, padding=0.05, visualize=False):
    """Convert point cloud to outline using OpenCV contour detection.
    
    Args:
        points: Input point cloud (N, 2) tensor or array
        img_size: Size of the canvas for processing
        padding: Padding as fraction of point cloud range
        visualize: If True, display the canvas and largest contour
    
    Returns:
        outline_points: Array of points forming the largest contour
    """
    points = points.cpu().numpy() if hasattr(points, 'cpu') else points
    min_xy = points.min()
    max_xy = points.max()
    range_xy = np.maximum(max_xy - min_xy, 1e-6)
    
    padded_min = min_xy - padding * range_xy
    padded_max = max_xy + padding * range_xy
    scale = img_size / (padded_max - padded_min)
    
    scaled_points = ((points - padded_min) * scale).astype(np.int32)
    canvas = np.zeros((img_size, img_size), dtype=np.uint8)
    for x, y in scaled_points:
        if 0 <= x < img_size and 0 <= y < img_size:
            canvas[y, x] = 255
    
    contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros((0, 2), dtype=np.float32)

    largest = max(contours, key=cv2.contourArea)
    contour_pixels = largest[:, 0, :].astype(np.float32)
    outline_points = contour_pixels / scale + padded_min
    
    if visualize:
        plt.figure(figsize=(10, 5))
        
        # Plot original canvas with points
        plt.subplot(121)
        plt.imshow(canvas, cmap='gray')
        plt.title('Point Cloud Canvas')
        plt.axis('off')
        
        # Plot canvas with largest contour highlighted
        contour_canvas = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        # contour_canvas[:, :, 1] = canvas  # Green channel for points
        cv2.drawContours(contour_canvas, [largest], -1, (255, 0, 255), 2)  # Magenta contour
        plt.subplot(122)
        plt.imshow(contour_canvas)
        plt.title('Largest Contour')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return outline_points

def _sample_points_on_polygon(poly, num_points, device='cpu', visualize=False):
    """Sample points uniformly along polygon edges.
    
    Args:
        poly: Tensor or array of shape (N, 2) representing polygon vertices.
        num_points: Number of points to sample.
        device: Device for PyTorch computations ('cpu' or 'cuda').
        visualize: If True, display visualizations of the process.
    
    Returns:
        Tensor of shape (num_points, 2) with sampled points.
    """
    # Convert input to tensor and ensure correct shape
    poly = torch.as_tensor(poly, device=device, dtype=torch.float32)
    if poly.dim() == 1:
        poly = poly.unsqueeze(0)  # Handle single point case
    if poly.shape[-1] != 2:
        raise ValueError(f"Expected poly to have shape (N, 2), got {poly.shape}")
    N = poly.shape[0]
    
    if N < 2:
        # Return repeated first point if N < 2
        return poly[0].repeat(num_points, 1) if N == 1 else torch.zeros((num_points, 2), device=device)
    
    # Compute edges and their lengths
    edges = poly[1:] - poly[:-1]
    lengths = torch.norm(edges, dim=1)
    perim = lengths.sum()
    if perim < 1e-9:
        return poly[0].repeat(num_points, 1)
    
    # Compute cumulative lengths
    cum_len = torch.cumsum(lengths, dim=0)
    
    # Sample points
    samples = []
    for _ in range(num_points):
        r = torch.rand(1, device=device) * perim
        idx = torch.searchsorted(cum_len, r)
        idx = min(idx, len(lengths) - 1)
        t = (r - (cum_len[idx - 1] if idx > 0 else 0.0)) / (lengths[idx] + 1e-12)
        start = poly[idx]
        end = poly[(idx + 1) % N]
        samples.append(start + t * (end - start))
    
    samples = torch.stack(samples).squeeze(1)
    
    if visualize:
        # Convert to numpy for plotting
        poly_np = poly.cpu().numpy()
        samples_np = samples.cpu().numpy()
        
        # Ensure samples_np is 2D
        if samples_np.ndim == 1:
            samples_np = samples_np.reshape(-1, 1)
        if samples_np.shape[1] != 2:
            print(f"Warning: samples_np has shape {samples_np.shape}, expected (:, 2)")
            return samples
        
        # Create figure with subplots
        plt.figure(figsize=(15, 10))

        # Subplot 1: Original Polygon
        plt.subplot(221)
        plt.plot(poly_np[:, 0], poly_np[:, 1], 'b-', label='Polygon')
        plt.plot(poly_np[:, 0], poly_np[:, 1], 'bo', label='Vertices')
        if N > 1:
            plt.plot([poly_np[-1, 0], poly_np[0, 0]], [poly_np[-1, 1], poly_np[0, 1]], 'b-')
        plt.title('Original Polygon')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True)
        plt.axis("equal")
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)

        # Subplot 2: Edges with Lengths
        plt.subplot(222)
        for i in range(N - 1):
            plt.plot(poly_np[i:i+2, 0], poly_np[i:i+2, 1], 'g-')
            mid = (poly_np[i] + poly_np[i+1]) / 2
            plt.text(mid[0], mid[1], f'{lengths[i]:.2f}', color='red')
        if N > 1:
            plt.plot([poly_np[-1, 0], poly_np[0, 0]], [poly_np[-1, 1], poly_np[0, 1]], 'g-')
        plt.title('Edges with Lengths')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.axis("equal")
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)

        # Subplot 3: Sampled Points
        plt.subplot(223)
        plt.plot(poly_np[:, 0], poly_np[:, 1], 'b-', alpha=0.3)
        if N > 1:
            plt.plot([poly_np[-1, 0], poly_np[0, 0]], [poly_np[-1, 1], poly_np[0, 1]], 'b-', alpha=0.3)
        plt.scatter(samples_np[:, 0], samples_np[:, 1], color='red', label='Sampled Points')
        plt.title('Sampled Points')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True)
        plt.axis("equal")
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)

        # Subplot 4: Combined View
        plt.subplot(224)
        plt.plot(poly_np[:, 0], poly_np[:, 1], 'b-', label='Polygon')
        if N > 1:
            plt.plot([poly_np[-1, 0], poly_np[0, 0]], [poly_np[-1, 1], poly_np[0, 1]], 'b-')
        plt.scatter(samples_np[:, 0], samples_np[:, 1], color='red', label='Sampled Points')
        plt.title('Polygon with Sampled Points')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True)
        plt.axis("equal")
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)

        plt.tight_layout()
        plt.show()

    
    return samples


def contour_intersection_area(contour1, contour2, visualize=True):
    poly1 = Polygon(np.round(contour1, 1)).boundary
    poly2 = Polygon(np.round(contour2, 1)).boundary

    if not poly1.is_valid or not poly2.is_valid:
        raise ValueError("One of the contours is not a valid polygon")

    union = poly1.union(poly2)
    intersection = poly1.intersection(poly2)

    union_length = union.length
    intersection_length = intersection.length

    if visualize:
        fig, ax = plt.subplots(figsize=(6,6))
        
        # Plot contour1
        x1, y1 = poly1.xy
        ax.plot(x1, y1, 'b-', label='Contour 1', linewidth=2)
        
        # Plot contour2
        x2, y2 = poly2.xy
        ax.plot(x2, y2, 'r-', label='Contour 2', linewidth=2)
        
        # Plot intersection
        def plot_geom(geom, color='g', label='Intersection', linewidth=3, label_shown=False):
            if geom.is_empty:
                return
            if geom.geom_type in ['LineString', 'LinearRing']:
                xi, yi = geom.xy
                ax.plot(xi, yi, color, label=label if not label_shown else None, linewidth=linewidth)
            elif geom.geom_type == 'MultiLineString':
                for i, line in enumerate(geom.geoms):
                    ax.plot(*line.xy, color, label=label if not label_shown and i==0 else None, linewidth=linewidth)
            elif geom.geom_type == 'GeometryCollection':
                first = not label_shown
                for g in geom.geoms:
                    plot_geom(g, color=color, label=label, linewidth=linewidth, label_shown=not first)
                    first = False
            # Points are ignored

        plot_geom(intersection)
        
        ax.set_aspect('equal')
        ax.legend()
        ax.set_title("Contour Intersection")
        plt.show()


    return {
        "union_length": union_length,
        "intersection_length": intersection_length,
        "contour1_length": poly1.length,
        "contour2_length": poly2.length
    }


def generate_sample(num_points, max_lines=1, noise_std=0.01, device='cpu', line_contribute_threshold=0.3, visualize=False):
    """Generate one sample: point cloud and symmetry lines."""
    num_lines = torch.randint(2, max_lines + 1, (1,)).item()
    n, c, d = _random_line(batch_size=num_lines, device=device)
    curves, reflected = _generate_polynomial_shape(n, c, d, num_curve_pts=5000, device=device, visualize=visualize)
    
    all_points = torch.cat([curves, reflected], dim=1)  # (batch_size, 2*num_curve_pts, 2)
    all_points = all_points.view(-1, 2)  # (num_lines * 2 * num_curve_pts, 2)
    outline = point_cloud_to_outline(all_points, visualize=visualize)
    keep_lines_idxs = []
    for i in range(num_lines):
        line_outline = point_cloud_to_outline(torch.cat([curves[i], reflected[i]], dim=0), visualize=visualize)
        inter_dict = contour_intersection_area(outline, line_outline, visualize=visualize)
        if inter_dict["intersection_length"] / inter_dict["contour2_length"] >= line_contribute_threshold:
            keep_lines_idxs.append(i)
    if outline.size == 0:
        return torch.zeros((num_points, 2), device=device), torch.zeros((0, 3), device=device)

    contour = outline
    contour = torch.tensor(contour, device=device)
    
    # Normalize
    mean = contour.mean(dim=0, keepdim=True)
    contour = contour - mean
    max_abs = torch.abs(contour).max()
    if max_abs > 0:
        contour = contour / max_abs
    
    # Transform lines into normalized space
    lines = torch.zeros((len(keep_lines_idxs), 3), device=device)
    for i, idx in enumerate(keep_lines_idxs):
        n_i = n[idx]
        c_i = c[idx]
        # c_prime = (c_i + torch.dot(n_i, mean.squeeze())) / max_abs if max_abs > 0 else c_i + torch.dot(n_i, mean.squeeze())
        c_prime = (c_i - torch.dot(n_i, mean.squeeze())) / max_abs
        lines[i] = torch.tensor([n_i[0], n_i[1], c_prime], device=device)
    
    # Sample points on contour
    contour_pts = _sample_points_on_polygon(contour, num_points, device=device, visualize=visualize)
    if noise_std > 0:
        contour_pts += torch.normal(mean=0.0, std=noise_std, size=contour_pts.shape, device=device)
    
    return contour_pts, lines


def generate_dataset(num_samples, points_per_sample, max_lines=1, noise_std=0.01, device='cpu', line_contribute_threshold=0.3, visualize=False):
    """Generate dataset of point clouds and symmetry lines."""
    data = []
    lines = []
    for _ in tqdm(range(num_samples)):
        pts, lns = generate_sample(points_per_sample, max_lines, noise_std, device, line_contribute_threshold=line_contribute_threshold, visualize=visualize)
        data.append(pts[:].cpu().numpy())
        lines.append(lns.cpu().numpy())
    
    return np.array(data, dtype=object), np.array(lines, dtype=object)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic symmetry dataset with PyTorch")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples")
    parser.add_argument("--points_per_sample", type=int, default=2048, help="Points per sample")
    parser.add_argument("--max_lines", type=int, default=2, help="Max symmetry lines")
    parser.add_argument("--noise", type=float, default=0.01, help="Gaussian noise std")
    parser.add_argument("--output", type=str, default="dataset.npz", help="Output file")
    parser.add_argument("--visualize", action="store_true", help="Visualize samples")
    parser.add_argument("--device", type=str, default="cpu" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--lc_thresh", type=float, default=0.3, help="Line contribute threshold")
    args = parser.parse_args()
    
    points, lines = generate_dataset(
        args.num_samples, args.points_per_sample, args.max_lines, args.noise, args.device, line_contribute_threshold=args.lc_thresh, visualize=args.visualize
    )
    np.savez(args.output, points=points, lines=lines)
    print(f"Dataset saved to {args.output}")
