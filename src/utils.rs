use chfft::RFft1D;

pub fn psd(x: &[f64]) -> Vec<f64> {
    let mut fft = RFft1D::new(x.len());
    let y = fft.forward(x);
    y.iter().map(|x| x.norm_sqr()).collect()
}

pub fn auto_fov_center(ra: &[f64], dec: &[f64]) -> (f64, f64) {
    let dec_mean = dec.iter().sum::<f64>() / dec.len() as f64;
    let cc = ra
        .iter()
        .map(|&a| (a.cos(), a.sin()))
        .fold((0., 0.), |x, y| (x.0 + y.0, x.1 + y.1));
    let ra_mean = f64::atan2(cc.1, cc.0);
    (ra_mean, dec_mean)
}
