use clap::{App, Arg};
use linear_solver::io::RawMM;
use num_traits::float::FloatConst;
use scorus::coordinates::sphcoord::SphCoord;
use simobs::gridder::Gridder;
use simobs::utils::auto_fov_center;
fn main() {
    let matches = App::new("generate ptr mat")
        .arg(
            Arg::with_name("radec")
                .short("d")
                .long("dir")
                .required(true)
                .takes_value(true)
                .value_name("pointing directions")
                .help("ra dec file"),
        )
        .arg(
            Arg::with_name("center")
                .short("c")
                .long("center")
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
                .long("psize")
                .required(true)
                .takes_value(true)
                .value_name("pixel size")
                .help("pixel size"),
        )
        .arg(
            Arg::with_name("pointing matrix output")
                .short("o")
                .long("out")
                .required(true)
                .takes_value(true)
                .value_name("out file name")
                .help("out pointing matrix file name"),
        )
        .arg(
            Arg::with_name("pixel list file name")
                .short("p")
                .long("plist")
                .required(true)
                .takes_value(true)
                .value_name("pixel file name")
                .help("pixel file name"),
        )
        .get_matches();

    let ptr = RawMM::<f64>::from_file(matches.value_of("radec").unwrap()).to_array2();
    //println!("{:?}", ptr.shape());

    let ptr = ptr / 180.0 * f64::PI();

    let ra = ptr.column(0).to_owned();
    let dec = ptr.column(1).to_owned();

    let sph_list: Vec<_> = ra
        .iter()
        .zip(dec.iter())
        .map(|(&r, &d)| SphCoord::new(f64::PI() / 2.0 - d, r))
        .collect();

    //println!("point({}, {}) # point=x color=red", center[0].0, center[0].1);

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
    
    let step = matches
        .value_of("pixel size")
        .unwrap()
        .parse::<f64>()
        .unwrap()
        .to_radians();
    let gridder = Gridder::new(fov_center.pol, fov_center.az, step, step);

    let (m, pix_idx) = gridder.get_ptr_matrix(&sph_list);

    RawMM::from_sparse(&m).to_file(matches.value_of("pointing matrix output").unwrap());

    RawMM::from_array2(pix_idx.view()).to_file(matches.value_of("pixel list file name").unwrap());
}
