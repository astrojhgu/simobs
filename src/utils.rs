use chfft::RFft1D;


pub fn psd(x: &[f64])->Vec<f64>{
    let mut fft=RFft1D::new(x.len());
    let y=fft.forward(x);
    y.iter().map(|x|{
        x.norm_sqr()
    }).collect()
}

