/// Orbit camera — matches the C++ Camera class behavior.

export class Camera {
    constructor() {
        this.target = [0, 0, 0];
        this.distance = 3.0;
        this.theta = 0.0;   // azimuth
        this.phi = 0.3;     // elevation
        this.fov = 45.0;    // degrees
        this.near = 0.01;
        this.far = 1000.0;
    }

    lookAt(center, distance) {
        this.target = [center[0], center[1], center[2]];
        this.distance = distance;
    }

    rotate(dx, dy) {
        this.theta -= dx * 0.01;
        this.phi += dy * 0.01;
        const limit = Math.PI / 2 - 0.01;
        this.phi = Math.max(-limit, Math.min(limit, this.phi));
    }

    zoom(delta) {
        this.distance *= Math.exp(-delta * 0.1);
        this.distance = Math.max(0.01, this.distance);
    }

    pan(dx, dy) {
        const scale = this.distance * 0.002;
        const right = [Math.cos(this.theta), 0, -Math.sin(this.theta)];
        const up = [0, 1, 0];
        for (let i = 0; i < 3; i++) {
            this.target[i] -= right[i] * dx * scale;
            this.target[i] += up[i] * dy * scale;
        }
    }

    position() {
        const cp = Math.cos(this.phi), sp = Math.sin(this.phi);
        const ct = Math.cos(this.theta), st = Math.sin(this.theta);
        return [
            this.target[0] + this.distance * cp * st,
            this.target[1] + this.distance * sp,
            this.target[2] + this.distance * cp * ct,
        ];
    }

    viewMatrix() {
        const eye = this.position();
        const fx = this.target[0] - eye[0];
        const fy = this.target[1] - eye[1];
        const fz = this.target[2] - eye[2];
        const fl = Math.sqrt(fx * fx + fy * fy + fz * fz);
        const fwd = [fx / fl, fy / fl, fz / fl];

        // right = forward x world_up
        let rx = fwd[1] * 0 - fwd[2] * 1;
        let ry = fwd[2] * 0 - fwd[0] * 0;
        let rz = fwd[0] * 1 - fwd[1] * 0;
        // world_up = (0,1,0), so: right = (fwd.y*0 - fwd.z*1, fwd.z*0 - fwd.x*0, fwd.x*1 - fwd.y*0)
        // Simplify:
        rx = -fwd[2];
        ry = 0;
        rz = fwd[0];
        const rl = Math.sqrt(rx * rx + ry * ry + rz * rz);
        rx /= rl; ry /= rl; rz /= rl;

        // up = right x forward
        const ux = ry * fwd[2] - rz * fwd[1];
        const uy = rz * fwd[0] - rx * fwd[2];
        const uz = rx * fwd[1] - ry * fwd[0];

        // Column-major for WebGL
        return new Float32Array([
            rx, ux, -fwd[0], 0,
            ry, uy, -fwd[1], 0,
            rz, uz, -fwd[2], 0,
            -(rx * eye[0] + ry * eye[1] + rz * eye[2]),
            -(ux * eye[0] + uy * eye[1] + uz * eye[2]),
            (fwd[0] * eye[0] + fwd[1] * eye[1] + fwd[2] * eye[2]),
            1,
        ]);
    }

    projectionMatrix(aspect) {
        const f = 1.0 / Math.tan((this.fov * Math.PI / 180) / 2);
        const nf = this.near - this.far;
        // Column-major
        return new Float32Array([
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (this.far + this.near) / nf, -1,
            0, 0, (2 * this.far * this.near) / nf, 0,
        ]);
    }
}
