use ndarray::Array2;
use scorus::coordinates::sphcoord;
use scorus::coordinates::vec2d;
use scorus::coordinates::vec3d;
use sprs::CsMat;
use std::collections::BTreeMap;

pub struct Gridder {
    center: sphcoord::SphCoord<f64>,
    center_vec: vec3d::Vec3d<f64>,
    vdpol: vec3d::Vec3d<f64>,
    vdaz: vec3d::Vec3d<f64>,
    step_x: f64,
    step_y: f64,
}

pub fn query_pixel(pm: &mut BTreeMap<(isize, isize), usize>, pix: (isize, isize)) -> usize {
    let n = pm.len();
    //eprintln!("{}", n);
    *pm.entry(pix).or_insert(n)
}

impl Gridder {
    pub fn new(central_pol: f64, central_az: f64, step_x: f64, step_y: f64) -> Gridder {
        let center = sphcoord::SphCoord::new(central_pol, central_az);
        let center_vec = vec3d::Vec3d::from_sph_coord(center);
        let vdaz = center.vdaz();
        let vdpol = center.vdpol();

        Gridder {
            center,
            center_vec,
            vdpol,
            vdaz,
            step_x,
            step_y,
        }
    }

    pub fn proj(&self, sph: sphcoord::SphCoord<f64>) -> vec2d::Vec2d<f64> {
        let ang = sph.angle_between(self.center);
        let diff_vec = vec3d::Vec3d::from(sph) - vec3d::Vec3d::from(self.center);
        let diff_vec = {
            let d = diff_vec - self.center_vec * self.center_vec.dot(diff_vec);
            let l = d.length();
            if l == 0.0 {
                d
            } else {
                d * (ang / l)
            }
        };
        let diff_vec_az = diff_vec.dot(self.vdaz);
        let diff_vec_pol = diff_vec.dot(self.vdpol);
        vec2d::Vec2d::new(diff_vec_pol, diff_vec_az)
    }

    pub fn deproj(&self, xy: vec2d::Vec2d<f64>) -> sphcoord::SphCoord<f64> {
        let angle = xy.length();
        let xy = {
            if angle == 0.0 {
                xy
            } else {
                xy.normalized() * angle.tan()
            }
        };
        //println!("{:?}", xy);
        sphcoord::SphCoord::from(self.center_vec + self.vdpol * xy.x + self.vdaz * xy.y)
    }

    pub fn grid_index(&self, sph: sphcoord::SphCoord<f64>) -> (isize, isize) {
        let p = self.proj(sph);
        let i = (p.x / self.step_x).round() as isize;
        let j = (p.y / self.step_y).round() as isize;
        (i, j)
    }

    pub fn grid_index_gaussian(&self, sph: sphcoord::SphCoord<f64>, rcut_rad: f64, sigma_rad: f64) -> (Vec<(isize, isize)>, Vec<f64>) {
        let p = self.proj(sph);
        let i0 = (p.x / self.step_x).round() as isize;
        let j0 = (p.y / self.step_y).round() as isize;

        let i_min = i0 - (rcut_rad / self.step_x).round() as isize;
        let j_min = j0 - (rcut_rad / self.step_y).round() as isize;

        let i_max = i0 + (rcut_rad / self.step_x).round() as isize;
        let j_max = j0 + (rcut_rad / self.step_y).round() as isize;

        let mut pix_list=Vec::new();
        let mut wgt_list=Vec::new();

        let mut total_wgt=0.0;
        for i in i_min ..= i_max{
            for j in j_min ..= j_max{
                let p1 = vec2d::Vec2d::new(p.x + (i-i0) as f64 * self.step_x, p.y + (j-j0) as f64 * self.step_y);
                let sph1 = self.deproj(p1);
                let r = sph.angle_between(sph1);
                if r < rcut_rad{
                    let wgt=(-r/2.0/sigma_rad).exp();
                    pix_list.push((i,j));
                    wgt_list.push(wgt);
                    total_wgt+=wgt;
                }
            }
        }

        for w in &mut wgt_list{
            *w/=total_wgt;
        }
        (pix_list, wgt_list)
    }

    pub fn get_ptr_matrix(
        &self,
        points: &[sphcoord::SphCoord<f64>],
    ) -> (CsMat<f64>, Array2<isize>) {
        let mut pix_map = BTreeMap::<(isize, isize), usize>::new();
        let mut ptr_idx = vec![0];
        let mut indices = vec![];
        let mut values = vec![];
        for &point in points.iter() {
            let pix = self.grid_index(point);
            let n = query_pixel(&mut pix_map, pix);
            ptr_idx.push(*ptr_idx.last().unwrap() + 1);
            indices.push(n);
            values.push(1.0_f64);
        }
        //let mut pixel_indices=vec![(0,0); pix_map.len()];
        let mut pixel_indices = Array2::zeros((pix_map.len(), 2));
        for (&k, &v) in pix_map.iter() {
            //pixel_indices[v]=k;
            pixel_indices[(v, 0)] = k.0;
            pixel_indices[(v, 1)] = k.1;
        }
        (
            CsMat::new((points.len(), pix_map.len()), ptr_idx, indices, values),
            pixel_indices,
        )
    }

    pub fn get_ptr_matrix_mo(
        &self,
        points: &[&[sphcoord::SphCoord<f64>]],
    ) -> (Vec<CsMat<f64>>, Array2<isize>) {
        let mut pix_map = BTreeMap::<(isize, isize), usize>::new();
        let mut ptr_idx = vec![vec![0]; points.len()];
        let mut indices = vec![vec![]; points.len()];
        let mut values = vec![vec![]; points.len()];
        for (i, points1) in points.iter().enumerate(){
            for &point in points1.iter() {
                let pix = self.grid_index(point);
                let n = query_pixel(&mut pix_map, pix);
                let m = *ptr_idx[i].last().unwrap() + 1;
                ptr_idx[i].push(m);
                indices[i].push(n);
                values[i].push(1.0_f64);
            }
        }
        //let mut pixel_indices=vec![(0,0); pix_map.len()];
        let mut pixel_indices = Array2::zeros((pix_map.len(), 2));
        for (&k, &v) in pix_map.iter() {
            //pixel_indices[v]=k;
            pixel_indices[(v, 0)] = k.0;
            pixel_indices[(v, 1)] = k.1;
        }
        (
            points.iter().zip(ptr_idx.into_iter().zip(indices.into_iter().zip(values.into_iter()))).map(|(i, (pi, (idx, v)))|{
                CsMat::new((i.len(), pix_map.len()), pi, idx, v)    
            }).collect()
            ,
            pixel_indices,
        )
    }

    pub fn get_ptr_matrix_mo_gaussian(
        &self,
        points: &[&[sphcoord::SphCoord<f64>]],
        rcut_rad: f64, 
        sigma_rad: f64,
    ) -> (Vec<CsMat<f64>>, Array2<isize>) {
        let mut pix_map = BTreeMap::<(isize, isize), usize>::new();
        let mut ptr_idx = vec![vec![0]; points.len()];
        let mut indices = vec![vec![]; points.len()];
        let mut values = vec![vec![]; points.len()];
        
        for (i, points1) in points.iter().enumerate(){
            for &point in points1.iter() {
                let (pix_list, wgt_list) = self.grid_index_gaussian(point, rcut_rad, sigma_rad);
                let m = *ptr_idx[i].last().unwrap() + pix_list.len();
                ptr_idx[i].push(m);
                //println!("{}", pix_list.len());
                    
                for (pix, w) in pix_list.into_iter().zip(wgt_list){
                    let n = query_pixel(&mut pix_map, pix);
                    indices[i].push(n);
                    values[i].push(w);
                }
            }
        }
        //let mut pixel_indices=vec![(0,0); pix_map.len()];
        let mut pixel_indices = Array2::zeros((pix_map.len(), 2));
        for (&k, &v) in pix_map.iter() {
            //pixel_indices[v]=k;
            pixel_indices[(v, 0)] = k.0;
            pixel_indices[(v, 1)] = k.1;
        }
        (
            points.iter().zip(ptr_idx.into_iter().zip(indices.into_iter().zip(values.into_iter()))).map(|(i, (pi, (idx, v)))|{
                CsMat::new((i.len(), pix_map.len()), pi, idx, v)    
            }).collect()
            ,
            pixel_indices,
        )
    }
}
