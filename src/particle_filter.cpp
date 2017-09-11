/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	//************* Set the number of particles. *************
	num_particles = 100;

	//************* Initialize all particles to first position (based on estimates of x, y, theta and their uncertainties from GPS) and all weights to 1. *************
	
	//Gaussian Distribution for x, y and theta
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	for (int i = 0; i < num_particles; ++i)
	{
		Particle p;

		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1;
		
		particles.push_back(p);			
	}
	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	for (int i = 0; i < num_particles; ++i)
	{
		//Add measurements to each particle
		particles[i].x = particles[i].x + (velocity/yaw_rate) * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
		particles[i].y = particles[i].y + (velocity/yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
		particles[i].theta = particles[i].theta + yaw_rate*delta_t;

		//Add random Gaussian noise
		normal_distribution<double> dist_x(particles[i].x, std_pos[0]);
		normal_distribution<double> dist_y(particles[i].y, std_pos[1]);
		normal_distribution<double> dist_theta(particles[i].theta, std_pos[2]);

		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for (int o_ix = 0; o_ix < observations.size(); ++o_ix)
	{
		double min_dist = numeric_limits<double>::max();
		int min_ix = -1;
		for (int p_ix = 0; p_ix < predicted.size(); ++p_ix)
		{
			double tmp_dist = dist(predicted[p_ix].x,predicted[p_ix].y, observations[o_ix].x, observations[o_ix].y);
			if(tmp_dist < min_dist){
				min_dist = tmp_dist;
				min_ix = p_ix;
			}
		}
		observations[o_ix].id = min_ix;
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	
	for (int pa_ix = 0; pa_ix < particles.size(); ++pa_ix)
	{
	 	Particle pa = particles[pa_ix];

	 	vector<LandmarkObs> predicted;

	 	//Find all landmarks in map within sensor range of a chosen Particle
	 	for (int ma_ix = 0; ma_ix < map_landmarks.landmark_list.size(); ++ma_ix)
	 	{
	 		//Pick next single landmark
	 		Map::single_landmark_s single_la = map_landmarks.landmark_list[ma_ix];

	 		//Calculate distance from particle to landmark and test if within sensor_range
	 		double distance = dist(pa.x, pa.y, single_la.x_f, single_la.y_f);

	 		if(distance <= sensor_range){
	 			LandmarkObs pred_single;
		        pred_single.id = single_la.id_i;
		        pred_single.x = single_la.x_f;
		        pred_single.y = single_la.y_f;
		        predicted.push_back(pred_single);
	 		}
		}

		//If no landmark in range of sensor_range -> set weight to zero
		if (predicted.size() == 0)
	    {
	       particles[pa_ix].weight = 0;
	    }
	    else
	    {
	    	//Transform observation from vehicle coordinates in map coordinates
	      	vector<LandmarkObs> observations_map;
	        observations_map.resize(observations.size());
	        for (int j = 0; j < observations_map.size(); j++)
	        {
	        	observations_map[j].x = pa.x + observations[j].x*cos(pa.theta) - observations[j].y*sin(pa.theta);
	        	observations_map[j].y = pa.y + observations[j].x*sin(pa.theta) + observations[j].y*cos(pa.theta);
	        }

	        //Find the predicted measurement that is closest to each observed measurement
	        dataAssociation(predicted, observations_map);
	        double pa_weight = 1;

	        //Calculate weights
	        for (int i = 0; i < observations_map.size(); i++)
	        {
	        	/*
	        	double tmp = 0;
	        	int p_ix = observations[i].id;
	        	double const x_o = observations_map[i].x;
	        	double const y_o = observations_map[i].x;
	        	double const x_p = predicted[p_ix].x;
	        	double const y_p = predicted[p_ix].x;

	        	tmp = (x_o - x_p)*(x_o - x_p)/(std_x*std_x) + (y_o - y_p)*(y_o - y_p)/(std_y*std_y);*/

	        	int p_ix = observations[i].id;
	        	const double obs_weight = multivariateGaussian(observations_map[i], predicted[p_ix], std_landmark);
	        	pa_weight *= obs_weight;
	        }
	        particles[pa_ix].weight = pa_weight;
	    }
	}
}

double ParticleFilter::multivariateGaussian(const LandmarkObs &obs_in_ws, const LandmarkObs &landmark_pt, double *std)
{
	// calculate multivariate Gaussian
	// e = ((x - mu_x) ** 2 / (2.0 * std_x ** 2)) + ((y - mu_y) ** 2 / (2.0 * std_y ** 2))
	// e = ((diff_x ** 2) / (2.0 * std_x ** 2)) + ((diff_y ** 2) / (2.0 * std_y ** 2))
	const double std_x = std[0];
	const double std_y = std[1];

	const double diff_x = obs_in_ws.x - landmark_pt.x;
	const double diff_y = obs_in_ws.y - landmark_pt.y;
	const double diff_x_2 = diff_x * diff_x;
	const double diff_y_2 = diff_y * diff_y;

	const double var_x = std_x * std_x;
	const double var_y = std_y * std_y;

	const double exponent = diff_x_2 / (2.0 * var_x) + diff_y_2 / (2.0 * var_y);
	const double exp_exponent = exp(-exponent);
			
	const double gauss_norm = 1.0 / (2.0 * M_PI * std_x * std_y);
	return gauss_norm * exp_exponent;
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    std::vector<double> weights;
    for (auto &p : particles)
        weights.push_back(p.weight);

    default_random_engine gen;
    std::discrete_distribution<> d(weights.begin(), weights.end());

    std::vector<Particle> newParticles;
    for (unsigned i = 0; i < particles.size(); i++)
        newParticles.push_back(particles[d(gen)]);

    particles = newParticles;
}


Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
