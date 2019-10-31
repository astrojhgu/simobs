use fitsimg::write_img;
use scorus::map_proj::mollweide::{proj, iproj};
use scorus::coordinates::Vec2d;
use scorus::coordinates::sphcoord::SphCoord;
use scorus::healpix::interp::natural_interp_ring;
use scorus::healpix::utils::npix2nside;
use simobs::scan::{dump_healpix_map, load_healpix_data_gal, project_ptr};
use simobs::gridder::Gridder;
use clap::{App, Arg};
use ndarray::Array2;
use linear_solver::io::RawMM;
use num_traits::float::FloatConst;
use simobs::noise::ColoredNoise;
fn main() {
    let data=load_healpix_data_gal("../allsky_408.fits", 128);
    let ptr=RawMM::<f64>::from_file("ptr.mtx").to_array2();
    //println!("{:?}", ptr.shape());
    
    let ptr=ptr/180.0*f64::PI();
    
    let ra=ptr.column(0).to_owned();
    let dec=ptr.column(1).to_owned();

    let sph_list:Vec<_>=ra.iter().zip(dec.iter()).map(|(&r, &d)|
    {
        SphCoord::new(f64::PI()/2.0-d, r)
    }
    ).collect();

    let mut cn=ColoredNoise::new(-1.0, 65536, 1.0);

    let nside=npix2nside(data.len());
    let tod:ndarray::Array1<_>=sph_list.iter().map(|&ptr|{
        natural_interp_ring(nside, &data, ptr)
        }).collect();

    RawMM::<f64>::from_array1(tod.view()).to_file("tod_pure.mtx");

    let noise:ndarray::Array1<_>=sph_list.iter().map(|_|{
        cn.get()
    }).collect();

    RawMM::<f64>::from_array1(noise.view()).to_file("noise.mtx");
    let noise_cov:ndarray::Array1<_>=cn.truncated_autocorr(tod.len()).iter().cloned().collect();

    RawMM::<f64>::from_array1(noise_cov.view()).to_file("noise_cov.mtx");

    let tod_with_noise=&tod+&noise;

    RawMM::from_array1(tod_with_noise.view()).to_file("tod_with_noise.mtx");


    dump_healpix_map(&data, "a.fits", 1024);

    let center=project_ptr(&[162.0_f64.to_radians()], &[3.5_f64.to_radians()], 1024);

    //println!("point({}, {}) # point=x color=red", center[0].0, center[0].1);

    let fov_center=SphCoord::new(f64::PI()/2.0-3.5_f64.to_radians(), 162.0_f64.to_radians());
    let step=0.025;
    let gridder=Gridder::new(fov_center.pol, fov_center.az, step, step);


    let (m, pix_idx)=gridder.get_ptr_matrix(&sph_list);

    RawMM::from_sparse(&m).to_file("ptr_matrix.mtx");

    RawMM::from_array2(pix_idx.view()).to_file("pixels.mtx");

    for r in 0..pix_idx.nrows(){
        let i=pix_idx[(r,0)];
        let j=pix_idx[(r,1)];
        let p1=Vec2d::new(i as f64*step, j as f64*step);

        let sp=gridder.deproj(p1);
        let ra1=sp.az;
        let dec1=(f64::PI()/2.0-sp.pol);
        //println!("{} {} {} {}", ra, dec, ra1, dec1);
        //println!("{} {} {} {}", ra, dec, );
        let p2=project_ptr(&[ra1],&[dec1], 1024);
        for(x,y) in p2{
            println!("point({} ,{}) # point=cross, color=white", x, y);
        }
        //println!("{} {}", i, j);

    }

    let height=1024.0;
    let width=height*2.0;
    let (x1, y1)=(1990.8233, 599.35505);
    let y = (y1 - height / 2.0) / height;
    let x = -(x1 - width / 2.0) / width * 2.0;

    let p = iproj(Vec2d::new(x, y)).unwrap();
    //println!("{:?}", p);
    //println!("{:}", 90.0-(p.pol as f64).to_degrees());
    //println!("{:}", (p.az as f64).to_degrees()+360.0);

    let (min_i, max_i)={
        let c=pix_idx.column(0);
        (*c.iter().min().unwrap(), *c.iter().max().unwrap())
    };

    let (min_j, max_j)={
        let c=pix_idx.column(1);
        (*c.iter().min().unwrap(), *c.iter().max().unwrap())
    };

    let height=(max_i-min_i+1) as usize;
    let width=(max_j-min_j+1) as usize;
    let mut img=Array2::<f64>::zeros((height, width));

    for &sph in sph_list.iter(){
        let (i, j)=gridder.grid_index(sph);
        img[(((i-min_i) as usize, (j-min_j) as usize))]=natural_interp_ring(nside, &data, sph);
    }

    write_img("answer.fits".to_string(), &img.into_dyn());
}

