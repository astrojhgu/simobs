use chfft::RFft1D;
use num_complex::Complex64;
use rand::distributions::Distribution;
use rand::rngs::ThreadRng;
use rand_distr::Normal;
pub struct ColoredNoise {
    pub normal_rand: Normal<f64>,
    pub rng: ThreadRng,
    pub buffer: std::collections::VecDeque<f64>,
    pub kernel: Vec<f64>,
    pub psd: Vec<f64>,
    pub autocorr: Vec<f64>,
}

impl ColoredNoise {
    pub fn new(beta: f64, length: usize, sigma: f64) -> ColoredNoise {
        let mut fft = RFft1D::new(length);
        let mut norm = 0.0;
        let kernel_f: Vec<_> = (0..=length / 2)
            .map(|i| {
                let x = if i == 0 {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::from((i as f64).powf(beta / 2.0))
                };
                norm += x.re.powi(2);
                x
            })
            .collect();

        let norm = sigma / (norm / (length / 2 + 1) as f64).sqrt();
        //println!("{:?}", norm as f64);

        let kernel_f: Vec<_> = kernel_f.iter().map(|&x| x * norm).collect();

        let normal_rand = Normal::new(0.0, 1.0).unwrap();
        let mut rng = rand::thread_rng();
        let kernel1 = fft.backward(&kernel_f);

        let psd: Vec<_> = kernel_f.iter().map(|&x| x.re.powi(2)).collect();

        let psd_complex: Vec<Complex64> = psd.iter().map(|&x| x.into()).collect();

        let autocorr = fft.backward(&psd_complex);
        //let normalization=sigma/autocorr[0].sqrt();

        let mut kernel = vec![0.0; length];
        for i in 0..length {
            if i < length / 2 {
                kernel[i] = kernel1[i + length / 2];
            } else {
                kernel[i] = kernel1[i - length / 2];
            }
        }

        let buffer: std::collections::VecDeque<_> =
            (0..length).map(|_| normal_rand.sample(&mut rng)).collect();
        ColoredNoise {
            normal_rand,
            rng,
            buffer,
            kernel,
            psd,
            autocorr,
        }
    }

    pub fn convolve(&self) -> f64 {
        self.buffer
            .iter()
            .zip(self.kernel.iter())
            .map(|(&x, &y)| x * y)
            .sum()
    }

    pub fn get(&mut self) -> f64 {
        let x = self.normal_rand.sample(&mut self.rng);
        self.buffer.push_back(x);
        self.buffer.pop_front();
        self.convolve()
    }

    pub fn truncated_autocorr(&self, length: usize) -> Vec<f64> {
        let mut result = vec![0.0; length];
        //for i in 0..length/2{
        for (i, r) in result.iter_mut().enumerate().take(length / 2) {
            if i<self.autocorr.len(){
                *r = self.autocorr[i];
            }else{
                *r = 0.0;
            }
        }
        for (i, r) in result.iter_mut().enumerate().take(length).skip(length / 2) {
            if self.autocorr.len() - length +i < self.autocorr.len(){
                *r = self.autocorr[self.autocorr.len() - length + i];
            }else{
                *r = 0.0;
            }
        }
        result
    }
}
