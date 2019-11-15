#![allow(clippy::many_single_char_names)]

use clap::{App, Arg};
use fitsimg::write_img;
use linear_solver::io::RawMM;
use ndarray::Array2;
use num_traits::float::FloatConst;
use scorus::coordinates::sphcoord::SphCoord;
use scorus::coordinates::Vec2d;
use scorus::healpix::interp::natural_interp_ring;
use scorus::healpix::utils::npix2nside;
use simobs::gridder::Gridder;
use simobs::noise::ColoredNoise;
use simobs::scan::{dump_healpix_map, load_healpix_data_gal, project_ptr};
use simobs::utils::auto_fov_center;

fn main() {
    let matches = App::new("simulate obs")
        .arg(
            Arg::with_name("sky template in healpix")
                .short("y")
                .long("sky")
                .required(true)
                .takes_value(true)
                .value_name("sky template file in ring healpix format")
                .help("sky template file"),
        )
        .arg(
            Arg::with_name("pointing dir")
                .short("p")
                .long("pointing")
                .required(true)
                .takes_value(true)
                .value_name("pointing file")
                .help("pointing file"),
        )
        .arg(
            Arg::with_name("pointing matrix")
                .short("m")
                .long("pm")
                .required(true)
                .takes_value(true)
                .value_name("pointing matrix file")
                .help("pointing matrix"),
        )
        .arg(
            Arg::with_name("out pure tod")
                .short("t")
                .long("tod")
                .required(true)
                .takes_value(true)
                .value_name("pure tod file")
                .help("pure tod file"),
        )
        .arg(
            Arg::with_name("out noisy tod")
                .short("o")
                .long("ntod")
                .required(true)
                .takes_value(true)
                .value_name("noisy tod file")
                .help("tod with noise added"),
        )
        .arg(
            Arg::with_name("noise realization")
                .short("n")
                .long("noise")
                .required(true)
                .takes_value(true)
                .value_name("noise file")
                .help("contains one noise realization"),
        )
        .arg(
            Arg::with_name("noise covariance")
                .short("c")
                .long("ncov")
                .required(true)
                .takes_value(true)
                .value_name("noise covariance")
                .help("noise covariance matrix"),
        )
        .arg(
            Arg::with_name("pixel list")
                .short("q")
                .long("pix")
                .required(true)
                .takes_value(true)
                .value_name("pixel list file")
                .help("pixel list"),
        )
        .arg(
            Arg::with_name("fov center")
                .short("f")
                .long("fovc")
                .required(false)
                .takes_value(true)
                .number_of_values(2)
                .value_delimiter(",")
                .value_names(&["ra", "dec"])
                .help("ra and dec for center in degree"),
        )
        .arg(
            Arg::with_name("pixel size")
                .short("s")
                .long("ps")
                .required(false)
                .takes_value(true)
                .value_name("pixel size")
                .help("pixel size")
        )
        .arg(
            Arg::with_name("noise sigma")
                .short("S")
                .long("sigma")
                .required(true)
                .takes_value(true)
                .value_name("sigma")
                .help("noise sigma")
        )
        .get_matches();

    let data = load_healpix_data_gal(matches.value_of("sky template in healpix").unwrap(), 128);
    let ptr = RawMM::<f64>::from_file(matches.value_of("pointing dir").unwrap()).to_array2();
    let sigma=matches.value_of("noise sigma").unwrap().parse::<f64>().unwrap();
    //println!("{:?}", ptr.shape());

    let ptr = ptr / 180.0 * f64::PI();

    let ra = ptr.column(0).to_owned();
    let dec = ptr.column(1).to_owned();

    let sph_list: Vec<_> = ra
        .iter()
        .zip(dec.iter())
        .map(|(&r, &d)| SphCoord::new(f64::PI() / 2.0 - d, r))
        .collect();

    let mut cn = ColoredNoise::new(-1.5, 65536, sigma);

    let nside = npix2nside(data.len());
    let tod: ndarray::Array1<_> = sph_list
        .iter()
        .map(|&ptr| natural_interp_ring(nside, &data, ptr))
        .collect();

    RawMM::<f64>::from_array1(tod.view()).to_file(matches.value_of("out pure tod").unwrap());

    let noise: ndarray::Array1<_> = sph_list.iter().map(|_| cn.get()).collect();

    RawMM::<f64>::from_array1(noise.view()).to_file(matches.value_of("noise realization").unwrap());
    let noise_cov: ndarray::Array1<_> = cn.truncated_autocorr(tod.len()).iter().cloned().collect();

    RawMM::<f64>::from_array1(noise_cov.view())
        .to_file(matches.value_of("noise covariance").unwrap());

    let tod_with_noise = &tod + &noise;

    RawMM::from_array1(tod_with_noise.view()).to_file(matches.value_of("out noisy tod").unwrap());

    dump_healpix_map(&data, "a.fits", 1024);

    let fov_center = if matches.is_present("fov center") {
        let fov_center_ra = matches
            .values_of("fov center")
            .unwrap()
            .nth(0)
            .unwrap()
            .parse::<f64>()
            .unwrap()
            .to_radians();
        let fov_center_dec = matches
            .values_of("fov center")
            .unwrap()
            .nth(1)
            .unwrap()
            .parse::<f64>()
            .unwrap()
            .to_radians();

        SphCoord::new(f64::PI() / 2.0 - fov_center_dec, fov_center_ra)
    } else {
        let (fov_center_ra, fov_center_dec) =
            auto_fov_center(ra.as_slice().unwrap(), dec.as_slice().unwrap());
        SphCoord::new(f64::PI() / 2.0 - fov_center_dec, fov_center_ra)
    };

    let step = 
        if matches.is_present("pixel size"){
            matches.value_of("pixel size").unwrap().parse::<f64>().unwrap().to_radians()
        }else{
            0.025
        };
    let gridder = Gridder::new(fov_center.pol, fov_center.az, step, step);

    let (m, pix_idx) = gridder.get_ptr_matrix(&sph_list);

    RawMM::from_sparse(&m).to_file(matches.value_of("pointing matrix").unwrap());

    RawMM::from_array2(pix_idx.view()).to_file(matches.value_of("pixel list").unwrap());

    for r in 0..pix_idx.nrows() {
        let i = pix_idx[(r, 0)];
        let j = pix_idx[(r, 1)];
        let p1 = Vec2d::new(i as f64 * step, j as f64 * step);

        let sp = gridder.deproj(p1);
        let ra1 = sp.az;
        let dec1 = f64::PI() / 2.0 - sp.pol;
        //println!("{} {} {} {}", ra, dec, ra1, dec1);
        //println!("{} {} {} {}", ra, dec, );
        let p2 = project_ptr(&[ra1], &[dec1], 1024);
        for (x, y) in p2 {
            println!("point({} ,{}) # point=cross, color=white", x, y);
        }
        //println!("{} {}", i, j);
    }

    //let p = iproj(Vec2d::new(x, y)).unwrap();
    //println!("{:?}", p);
    //println!("{:}", 90.0-(p.pol as f64).to_degrees());
    //println!("{:}", (p.az as f64).to_degrees()+360.0);

    let (min_i, max_i) = {
        let c = pix_idx.column(0);
        (*c.iter().min().unwrap(), *c.iter().max().unwrap())
    };

    let (min_j, max_j) = {
        let c = pix_idx.column(1);
        (*c.iter().min().unwrap(), *c.iter().max().unwrap())
    };

    let height = (max_i - min_i + 1) as usize;
    let width = (max_j - min_j + 1) as usize;
    let mut img = Array2::<f64>::zeros((height, width));

    for &sph in sph_list.iter() {
        let (i, j) = gridder.grid_index(sph);
        img[((i - min_i) as usize, (j - min_j) as usize)] = natural_interp_ring(nside, &data, sph);
    }

    write_img("answer.fits".to_string(), &img.into_dyn()).unwrap();
}
