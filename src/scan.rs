use num_traits::float::FloatConst;
use ndarray::Array2;
use scorus::healpix;
use scorus::coordinates::SphCoord;
use scorus::map_proj::mollweide::{proj, iproj};
use scorus::coordinates::Vec2d;
use scorus::healpix::interp::natural_interp_ring;
use scorus::healpix::pix::pix2ang_ring;
use scorus::healpix::utils::{npix2nside, nside2npix};

use astroalgo::eqpoint::EqPoint;
use astroalgo::galpoint::GalPoint;
use astroalgo::quant::HasValue;
use astroalgo::quant::{Angle, Epoch};


use fitsimg::fitsio::FitsFile;

use fitsimg::write_img;


pub fn gal2eq_ring(data_in: &[f64]) -> Vec<f64> {
    let frac_pi_2 = f64::FRAC_PI_2();
    let nside = npix2nside(data_in.len());
    let mut result = vec![0.0; data_in.len()];
    for i in 0..data_in.len() {
        let ptr = pix2ang_ring(nside, i);
        let ra = ptr.az;
        let dec = frac_pi_2 - ptr.pol;
        let eqp = EqPoint {
            ra: Angle(ra),
            dec: Angle(dec),
        }.at_epoch(Epoch(2000.0));
        let gal = GalPoint::from(eqp);
        let ptr_gal = SphCoord::new(frac_pi_2 - gal.b.v(), gal.l.v());
        result[i] = natural_interp_ring(nside, data_in, ptr_gal);
    }
    result
}


pub fn eq2gal_ring(data_in: &[f64]) -> Vec<f64> {
    let frac_pi_2 = f64::FRAC_PI_2();
    let nside = npix2nside(data_in.len());
    let mut result = vec![0.0; data_in.len()];
    for i in 0..data_in.len() {
        let ptr = pix2ang_ring(nside, i);
        let l = ptr.az;
        let b = frac_pi_2 - ptr.pol;
        let gal = GalPoint {
            l: Angle(l),
            b: Angle(b),
        };
        let eqp = gal.to_eqpoint(Epoch(2000.0));
        let ptr_eqp = SphCoord::new(frac_pi_2 - eqp.dec.v(), eqp.ra.v());
        result[i] = natural_interp_ring(nside, data_in, ptr_eqp);
    }
    result
}


pub fn load_healpix_data_gal(name: &str, nside: usize) -> Vec<f64> {
    let mut fptr = FitsFile::open(name).unwrap();
    let hdu = fptr.hdu(1);
    let data: Vec<f64> = hdu
        .and_then(|hdu| hdu.read_col(&mut fptr, "TEMPERATURE"))
        .unwrap();
    //let hdu=fptr.hdu(1);
    //let ordering:String=hdu.and_then(|hdu|{hdu.read_key(&mut fptr, "ORDERING")}).unwrap();

    let data: Vec<f64> = {
        let nside_raw = npix2nside(data.len());
        gal2eq_ring(
            &(0..nside2npix(nside))
                .map(|i| {
                    let ptr = pix2ang_ring(nside, i);
                    natural_interp_ring(nside_raw, &data[..], ptr)
                }).collect::<Vec<f64>>()[..],
        )
    };
    data
}


pub fn dump_healpix_map(data: &[f64], name: &str, height: usize) {
    let width = height as isize * 2;
    let mut mx = Array2::<f64>::zeros((height as usize, width as usize)).into_dyn();

    let nside = npix2nside(data.len());

    for i in 0..height as isize {
        let y = (i - height as isize / 2) as f64 / height as f64;
        for j in 0..width as isize {
            let x = -(j - width / 2) as f64 / width as f64 * 2.0;
            let p = iproj(Vec2d::new(x, y));
            if let Some(p) = p {
                //let eq=EqPointAtEpoch{ra:Angle(p.az), dec:Angle(pi/2.0-p.pol), epoch:Epoch(2000.0)};
                //let gal=GalPoint::from(eq);
                //let p1=SphCoord::new( pi/2.0-gal.b.0, gal.l.0);
                let v = natural_interp_ring(nside, data, p);
                mx[[i as usize, j as usize]] = v;
            }
        }
    }
    write_img(name.to_string(), &mx).unwrap();
}

pub fn project_ptr(ra: &[f64], dec: &[f64], height: usize)->Vec<(f64, f64)>{
    let frac_pi_2 = f64::FRAC_PI_2();
    let width=height*2;
    ra.iter().zip(dec.iter()).map(|(&r, &d)|{
        let ptr=SphCoord::new(frac_pi_2-d, r);
        let Vec2d{x, y}=proj(ptr);
        //println!("{:?} {:?}", x, y);
        let mut y1=y*height as f64+height as f64/2.0;
        let mut x1=width as f64/2.0-x/2.0*width as f64;
        if x1>width as f64{
            x1-=width as f64;
        }
        if x1<0.0{
            x1+=width as f64;
        }
        if y1>height as f64{
            y1-=height as f64;
        }
        if y1<0.0{
            y1+=height as f64;
        }
        (x1,y1)
    }).collect()
}

pub fn gen_tod(data: &[f64], ra: &[f64], dec: &[f64])->Vec<f64>{
    let frac_pi_2 = f64::FRAC_PI_2();
    let nside=npix2nside(data.len());
    ra.iter().zip(dec.iter()).map(|(&r, &d)|{
        let pol=frac_pi_2-d;
        let az=r;
        let ptg=SphCoord::new(pol, az);
        let v=natural_interp_ring(nside, data, ptg);
        v
    }).collect()
}

