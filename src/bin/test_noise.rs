use linear_solver::io::RawMM;
use simobs::noise::ColoredNoise;
pub fn main() {
    let mut cn = ColoredNoise::new(-1.0, 65536, 1.0);
    //let mut psd_result=vec![0.0; length/2+1];

    let noise: ndarray::Array1<_> = (0..65536 * 8).map(|_| cn.get()).collect();
    RawMM::from_array1(noise.view()).to_file("noise.mtx");
    let autocorr = ndarray::Array1::from(cn.autocorr);

    RawMM::from_array1(autocorr.view()).to_file("noise_cov.mtx");

    let psd: ndarray::Array1<_> = cn.psd.iter().cloned().collect();
    RawMM::from_array1(psd.view()).to_file("noise_psd.mtx");
}
