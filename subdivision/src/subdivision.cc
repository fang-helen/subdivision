#include "subdivision.h"
#include <iostream>
#include <math.h>
#include <fstream>
#include <unordered_map>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/io.hpp>

namespace std 
{
	// hash function for vec2s (edges) - N x N -> N
	// https://stackoverflow.com/questions/919612/mapping-two-integers-to-one-in-a-unique-and-deterministic-way
	template <>
	struct hash<glm::uvec2>
	{
		std::size_t operator()(const glm::uvec2& edge) const
		{
			int a = edge[0];
			int b = edge[1];
			unsigned long A = (unsigned long)(a >= 0 ? 2 * (long)a : -2 * (long)a - 1);
			unsigned long B = (unsigned long)(b >= 0 ? 2 * (long)b : -2 * (long)b - 1);
			long C = (long)((A >= B ? A * A + A + B : A + B * B) / 2);
			return a < 0 && b < 0 || a >= 0 && b >= 0 ? C : -C - 1;
		}
	};
};

glm::uvec2 make_edge(std::vector<unsigned> cur_face, unsigned j);
glm::uvec2 find_edge(std::vector<unsigned> cur_face, glm::uvec2 edge);
unsigned find_vertex(std::vector<unsigned> cur_face, unsigned vert);
glm::vec3 compute_normal(glm::vec4 v0, glm::vec4 v1, glm::vec4 v2);
float compute_angle(glm::vec4 anchor, glm::vec4 center, glm::vec4 v, glm::vec3 n);

Mesh::Mesh()
{
	// default constructor, makes a cube
	subdivision_type = scheme::DOO_SABIN;

	initialize();
}

Mesh::Mesh(char *fn)
{
	filename_ = (char*)malloc(strlen(fn) + 1);
	if (filename_ == nullptr) {
		std::cout << "error allocating memory to save file name." << std::endl;
		return;
	}
	strcpy(filename_, fn);


	// default to catmull
	subdivision_type = scheme::CATMULL;

	initialize();
}

Mesh::~Mesh()
{
	free(filename_);
}

void Mesh::initialize() {
	if (filename_ == nullptr) {
		initialize_cube();
	}
	else {
		load_from_file();
	}
	std::cout << "Mesh initialized to use " << scheme_name() << " subdivision" << std::endl;
}

// initialize Mesh to a cube
void Mesh::initialize_cube() {
	obj_vertices.clear();
	faces.clear();

	obj_vertices.push_back(glm::vec4(-0.5f, 0.5f, -0.5f, 1.0f)); // upper back left
	obj_vertices.push_back(glm::vec4(0.5f, 0.5f, -0.5f, 1.0f)); // upper back right
	obj_vertices.push_back(glm::vec4(0.5f, -0.5f, -0.5f, 1.0f)); // lower back right
	obj_vertices.push_back(glm::vec4(-0.5f, -0.5f, -0.5f, 1.0f)); // lower back left

	obj_vertices.push_back(glm::vec4(-0.5f, 0.5f, 0.5f, 1.0f)); // upper front left
	obj_vertices.push_back(glm::vec4(0.5f, 0.5f, 0.5f, 1.0f)); // upper front right
	obj_vertices.push_back(glm::vec4(0.5f, -0.5f, 0.5f, 1.0f)); // lower front right
	obj_vertices.push_back(glm::vec4(-0.5f, -0.5f, 0.5f, 1.0f)); // lower front left

	// make sure faces are counterclockwise
	faces.push_back({ 0, 1, 2, 3 }); // back
	faces.push_back({7, 6, 5, 4 }); // front
	faces.push_back({ 4, 5, 1, 0 }); // top
	faces.push_back({ 3, 2, 6, 7 }); // bottom
	faces.push_back({ 3, 7, 4, 0 }); // left
	faces.push_back({ 1, 5, 6, 2 }); // right

	dirty_ = true;
}

// (re)load mesh from obj
void Mesh::load_from_file() {
	if (filename_ == nullptr) {
		return;
	}

	obj_vertices.clear();
	faces.clear();

	std::ifstream input_file;
	input_file.open(filename_);
	if (input_file.is_open()) {

		std::string line;
		while (std::getline(input_file, line))
		{
			if (line.length() >= 2) {
				if (line[0] == 'v' && line[1] == ' ') {
					// vertex line
					std::vector<float> components;

					// navigate to first non-space pos
					size_t start = 1;
					while (start < line.length() && line[start] == ' ') {
						start++;
					}

					while (start < line.length()) {
						size_t index = start;
						// search for ' ' delimiters
						while (index < line.length()) {
							if (line[index] == ' ' || index == line.length() - 1) {
								components.push_back(std::stof(line.substr(start, index - start + 1)));
								start = index + 1;
								break;
							}
							index++;
						}
					}

					if (components.size() == 3) {
						obj_vertices.push_back(glm::vec4(components[0], components[1], components[2], 1.0f));
					}
					else {
						std::cout << "invalid input vertex with " << components.size() << " components, " << line << std::endl;
					}

				}
				else if (line[0] == 'f' && line[1] == ' ') {
					// face line
					std::vector<unsigned int> components;

					// navigate to first non-space pos
					size_t start = 1;
					while (start < line.length() && line[start] == ' ') {
						start++;
					}
					while (start < line.length()) {
						size_t index = start;
						// search for ' ' delimiters
						while (index < line.length()) {
							if (line[index] == ' ' || index == line.length() - 1) {
								// depending on if file uses '/' or ' ' delimiters for faces
								if (line.find("/") != std::string::npos) {
									// search for '/' delimiters
									for (int i = start; i <= index; i++) {
										if (line[i] == '/') {
											// we only care about 1st number in triplet right now
											// make sure to convert to 0-indexing
											components.push_back(std::stoul(line.substr(start, i - start)) - 1);
											break;
										}
									}
								}
								else {
									// make sure to convert to 0-indexing
									components.push_back(std::stoul(line.substr(start, index - start + 1)) - 1);
								}

								start = index + 1;
								break;
							}
							index++;
						}
					}
					faces.push_back(components);
				}
			}
		}
		input_file.close();
	}
	else {
		printf("Unable to load mesh from %s\n", filename_);
	}

	// use default cube if something went wrong
	if (obj_vertices.size() == 0 || faces.size() == 0) {
		std::cout << "Something went wrong... initializing default cube" << std::endl;

		free(filename_);
		filename_ = nullptr;
		initialize_cube();
	}
	else {
		printf("Initialized mesh from %s\n", filename_);

		dirty_ = true;
	}
}

void Mesh::set_clean()
{
	dirty_ = false;
}

bool Mesh::is_dirty() const
{
	return dirty_;
}

// switch up the subdivision scheme
void Mesh::cycle_scheme () {
	int prev_scheme = static_cast<int>(subdivision_type);
	subdivision_type = static_cast<scheme>(prev_scheme + 1);
	if (subdivision_type == scheme::END) {
		subdivision_type = scheme::CATMULL;
	}
	std::cout << "New subdivision scheme = " << scheme_name() << std::endl;
}

// wrapper function, perform the correct subdivision based on Mesh config
void Mesh::subdivide() {

	std::cout << "Performing 1 round of " << scheme_name() << " subdivision." << std::endl;

	if (subdivision_type == scheme::CATMULL) {
		catmull_clark();
	}
	else if (subdivision_type == scheme::LOOP) {
		loop();
	}
	else if (subdivision_type == scheme::DOO_SABIN) {
		doo_sabin();
	}

	dirty_ = true;
}

// macro to return string corresponding to subdivision scheme name
std::string Mesh::scheme_name() {
	if (subdivision_type == scheme::CATMULL) {
		return "catmull-clark";
	}
	if (subdivision_type == scheme::LOOP) {
		return "loop";
	}
	if (subdivision_type == scheme::DOO_SABIN) {
		return "doo-sabin";
	}

	// default to catmull
	subdivision_type == scheme::CATMULL;
	return "catmull-clark";
}

// macro to get shared vertex between 2 edges
unsigned get_common_index(glm::uvec2 edge1, glm::uvec2 edge2) {
	if (edge1[0] == edge2[0] || edge1[0] == edge2[1]) {
		return edge1[0];
	}
	else if (edge1[1] == edge2[0] || edge1[1] == edge2[1]) {
		return edge1[1];
	}
	return -1;
}

// performs 1 subdivision iteration using loop scheme
void Mesh::loop() {
	/*
	https://graphics.stanford.edu/~mdfisher/subdivision.html
	http://www.cs.cmu.edu/afs/cs/academic/class/15462-s14/www/lec_slides/Subdivision.pdf

	odd vertices (recomputed positions of newly created verts) -
		- interior - v = 3.0/8.0 * (a + b)+ 1.0/8.0 * (c + d)
		- boundary - v = 1.0/2.0 * (a + b)
	even vertices (recomputed positions of original verts) -
		- interior - v = v * (1 - k*BETA) + (sum of all k neighbor vertices) * BETA
			- if k > 3, beta = 3/(8 * k)
			- if k = 3, beta = 3/16
		- exterior - v = 1.0/8.0 * (a + b) + 3.0/4.0 * v

	*/

	// initialize vertex adjacency counter for even vertices
	std::vector<std::vector<int>> adj_verts;
	for (int i = 0; i < obj_vertices.size(); i++) {
		std::vector<int> adj;
		adj_verts.push_back(adj);
	}

	std::vector<glm::uvec2> edges;			// all unique edges
	std::vector<glm::vec4> edge_points;		// edge points, same size as edges

	std::unordered_map<glm::uvec2, unsigned int> edges_to_faces;		// map edge:unused_vert from face 1
	std::unordered_map<glm::uvec2, unsigned int> edges_to_index;		// to help index in edge vectors

	int double_count = 0;
	// compute odd vertices and build up adjacency information
	for (int i = 0; i < faces.size(); i++) {
		std::vector<unsigned int> cur_face = faces[i];
		if (cur_face.size() > 3) {
			// convert to tris and handle the rest later
			std::vector<std::vector<unsigned int>> tris = convert_tris(cur_face);
			for (int j = 1; j < tris.size(); j++) {
				faces.push_back(tris[j]);
			}
		}

		// edge points = average of the two neighbouring face points and two original endpoints
		for (int j = 0; j < 3; j++) {
			int unused_vert = cur_face[(j + 2) % 3];
			glm::uvec2 edge = glm::uvec2(cur_face[j], cur_face[(j + 1) % 3]);
			if (edge[0] > edge[1]) {
				edge = glm::uvec2(edge[1], edge[0]);
			}

			// use hashmaps to keep track of face-edge relationship
			if (edges_to_faces.find(edge) == edges_to_faces.end()) {
				// first occurrence of edge, save edge index
				edges.push_back(edge);
				edges_to_index[edge] = edges.size() - 1;
				// also save vert connectedness info
				adj_verts[edge[0]].push_back(edge[1]);
				adj_verts[edge[1]].push_back(edge[0]);

				// save as a "boundary" odd vertex
				edges_to_faces[edge] = unused_vert;
				glm::vec4 edge_point = 0.5f * (obj_vertices[edge[0]] + obj_vertices[edge[1]]);
				edge_points.push_back(edge_point);

			}
			else if (edges_to_faces[edge] >= 0) {
				// found the second face adjacent to this edge
				glm::vec4 a = obj_vertices[edge[0]];
				glm::vec4 b = obj_vertices[edge[1]];
				glm::vec4 c = obj_vertices[edges_to_faces[edge]];
				glm::vec4 d = obj_vertices[unused_vert];

				// compute "interior" odd vertex
				glm::vec4 edge_pt = (3.0f / 8.0f) * (a + b) + (1.0f / 8.0f) * (c + d);

				// override the "boundary" odd vertex
				edge_points[edges_to_index[edge]] = edge_pt;
				edges_to_faces[edge] = -1; // for debugging

				double_count++;
			}
			else {
				// > 2 faces adjacent to this edge
				// should not reach this line
				std::cout << "error - edge " << edge << " adjacent to >2 faces" << std::endl;
			}
		}
	}
	// compute the even vertices
	for (int i = 0; i < obj_vertices.size(); i++) {
		std::vector<int> cur_neighbors = adj_verts[i];
		int k = cur_neighbors.size();
		glm::vec4 v = obj_vertices[i];
		if (k > 2) {
			// interior vertex
			float beta = (k > 3) ? 3.0f / (8.0f * k) : 3.0f / 16.0f;

			glm::vec4 sum_neighbors = glm::vec4(0);
			for (int j = 0; j < k; j++) {
				sum_neighbors += obj_vertices[cur_neighbors[j]];
			}
			v = v * (1.0f - k * beta) + beta * sum_neighbors;

		}
		else if (k == 2) {
			// boundary vertex
			glm::vec4 a = obj_vertices[cur_neighbors[0]];
			glm::vec4 b = obj_vertices[cur_neighbors[1]];
			v = (1.0f / 8.0f) * (a + b) + (3.0f / 4.0f) * v;
		}
		else {
			// this should not happen
			std::cout << "degenerate tri? k = " << k << std::endl;
		}

		// move the recomputed vertex
		obj_vertices[i] = v;
	}

	int num_even_verts = obj_vertices.size();
	// add the newly created vertices to the mesh
	for (int i = 0; i < edge_points.size(); i++) {
		obj_vertices.push_back(edge_points[i]);
	}

	// rebuild faces
	std::vector<std::vector<unsigned>> new_tris;
	for (int i = 0; i < faces.size(); i++) {
		std::vector<unsigned> cur_face = faces[i];

		std::vector<glm::uvec2> cur_edges;
		// reconstruct face edges
		for (int j = 0; j < 3; j++) {
			glm::uvec2 edge = glm::uvec2(cur_face[j], cur_face[(j + 1) % 3]);
			if (edge[0] > edge[1]) {
				edge = glm::uvec2(edge[1], edge[0]);
			}
			cur_edges.push_back(edge);
		}

		// each tri gets split into 4 tris

		// 3 tris: 1 even vertex + 2 odd vertices
		for (int j = 0; j < cur_edges.size(); j++) {
			glm::uvec2 edge1 = cur_edges[j];
			glm::uvec2 edge2 = cur_edges[(j + 1) % 3];

			std::vector<unsigned> new_face = {
				edges_to_index[edge1] + num_even_verts,
				get_common_index(edge1, edge2),
				edges_to_index[edge2] + num_even_verts };
			new_tris.push_back(new_face);
		}

		// 1 tri: 3 odd vertices
		std::vector<unsigned> new_face = {
			edges_to_index[cur_edges[0]] + num_even_verts,
			edges_to_index[cur_edges[1]] + num_even_verts,
			edges_to_index[cur_edges[2]] + num_even_verts };
		new_tris.push_back(new_face);
	}

	faces.clear();
	for (int i = 0; i < new_tris.size(); i++) {
		faces.push_back(new_tris[i]);
	}
}

// performs 1 subdivision iteration using catmull clark scheme
void Mesh::catmull_clark()
{
	// data structures to record relationships

	std::vector<glm::vec4> face_points;
	std::vector<glm::uvec2> edges;								// all unique edges
	std::unordered_map<glm::uvec2, glm::vec4> edges_to_points;  // map edges to edge points
	std::unordered_map<glm::uvec2, unsigned int> edges_to_index;			// to help index in edges vector

	std::vector<glm::vec4> F;									// sum face points adjacent to orig points
	std::vector<float> F_counts;								// num face points adjacent to orig points
	std::vector<glm::vec4> R;									// sum edge midpoints adjacent to orig points
	std::vector<float> R_counts;								// num edge midpoints adjacent to orig points

	for (int i = 0; i < obj_vertices.size(); i++) {
		F.push_back(glm::vec4(0));
		F_counts.push_back(0);
		R.push_back(glm::vec4(0));
		R_counts.push_back(0);
	}

	for (int i = 0; i < faces.size(); i++) {
		std::vector<unsigned int> cur_face = faces[i];
		int verts_per_face = cur_face.size();

		// face point = average of all original points for the respective face
		glm::vec4 face_point = glm::vec4(0);
		for (int j = 0; j < verts_per_face; j++) {
			face_point += obj_vertices[cur_face[j]];
		}
		face_point /= float(verts_per_face);
		face_points.push_back(face_point);

		// save to calculate F (face point) averages for each point
		for (int j = 0; j < verts_per_face; j++) {
			F[cur_face[j]] += face_point;
			F_counts[cur_face[j]] += 1;
		}

		// edge points = average of the two neighbouring face points and two original endpoints
		for (int j = 0; j < verts_per_face; j++) {
			glm::uvec2 edge = make_edge(cur_face, j);

			// use hashmaps to keep track of face-edge relationship
			if (edges_to_points.find(edge) == edges_to_points.end()) {
				// save 1st face point for this edge
				edges.push_back(edge);

				// negative w - keep track of which edges have been paired
				edges_to_points[edge] = glm::vec4(face_point[0], face_point[1], face_point[2], -1.0f);
				edges_to_index[edge] = edges.size() - 1;
			}
			else if (edges_to_points[edge][3] < 0) {
				// found the other face point for this edge
				glm::vec4 edge_pt =
					(edges_to_points[edge] +
						face_point +
						obj_vertices[edge[0]] +
						obj_vertices[edge[1]]) / 4.0f;
				edge_pt[3] = 1.0f;
				edges_to_points[edge] = edge_pt;
			}
			else {
				// > 2 faces adjacent to this edge
				// should not reach this line
				std::cout << "error - edge " << edge << " adjacent to >2 faces" << std::endl;
			}
		}
	}

	// find points for calculating R (edge midpoints)
	for (int i = 0; i < edges.size(); i++) {
		glm::uvec2 cur_edge = edges[i];
		unsigned int endpoint1 = cur_edge[0];
		unsigned int endpoint2 = cur_edge[1];

		glm::vec4 edge_mid = (obj_vertices[endpoint1] + obj_vertices[endpoint2]) / 2.0f;

		// save to calculate R averages
		R[endpoint1] += edge_mid;
		R_counts[endpoint1] += 1;

		R[endpoint2] += edge_mid;
		R_counts[endpoint2] += 1;
	}

	// move each original point P to new point (F + 2R + (n - 3)P) / n 
	for (int i = 0; i < obj_vertices.size(); i++) {
		glm::vec4 face_average = F[i] / F_counts[i];
		glm::vec4 edge_average = R[i] / R_counts[i];

		// f_counts should be == r_counts, but might not be if boundary

		float n = std::max(F_counts[i], R_counts[i]);
		glm::vec4 new_point = (face_average + 2.0f * edge_average + (n - 3.0f) * obj_vertices[i]) / n;
		new_point[3] = 1.0f;

		// move the original point
		obj_vertices[i] = new_point;
	}

	// add the newly created face and edge points to list of vertices
	int num_verts = obj_vertices.size();
	for (int i = 0; i < face_points.size(); i++) {
		obj_vertices.push_back(face_points[i]);
	}
	for (int i = 0; i < edges.size(); i++) {
		glm::vec4 edge_vert = edges_to_points[edges[i]];

		// hack boundary edges - only adjacent to 1 face
		edge_vert[3] = 1.0f;
		edges_to_points[edges[i]] = edge_vert;
		obj_vertices.push_back(edge_vert);
	}

	// assemble the new faces using moved points, edge points, face points
	std::vector<std::vector<unsigned int>> quads;
	for (int i = 0; i < faces.size(); i++) {
		std::vector<unsigned int> cur_face = faces[i];
		int verts_per_face = cur_face.size();

		// index offsets - 
		//	face point index = num_verts + i
		//	edge point index = num_verts + face_points.size() + indexOf(edge)

		// reconstruct edges for this face and get edge point indices
		std::vector<unsigned> cur_edge_indices;
		for (int j = 0; j < verts_per_face; j++) {
			glm::uvec2 edge = make_edge(cur_face, j);
			cur_edge_indices.push_back(num_verts + face_points.size() + edges_to_index[edge]);

		}
		unsigned int face_index = num_verts + i;

		for (int j = 0; j < verts_per_face; j++) {
			// each quad face becomes 4 new quad faces --
			//	vert1, edge 1, face, edge 4
			//	vert2, edge2, face, edge 1
			//	...
			std::vector<unsigned int> new_face =
			{ 
				cur_face[j], 
				cur_edge_indices[j], 
				face_index, 
				cur_edge_indices[(j + verts_per_face - 1) % verts_per_face]
			};
			
			quads.push_back(new_face);
		}
	}

	// replace the mesh faces
	faces.clear();
	for (int i = 0; i < quads.size(); i++) {
		faces.push_back(quads[i]);
	}
}

// perform 1 subdivision iteration using doo-sabin scheme
void Mesh::doo_sabin() {
	std::vector<glm::uvec2> edges;
	std::unordered_map<glm::uvec2, unsigned int> edges_to_index; // indices of edges in edge vector

	std::vector<glm::vec4> new_face_points;
	std::vector<std::vector<unsigned>> new_faces;
	std::vector<unsigned> new_indices;

	std::vector<std::vector<unsigned>> vertices_to_faces;	// for each vert, save adj faces
	std::vector<glm::vec3> approx_vert_normals;				// for each vert, save avg normal of adj faces
	std::vector<glm::uvec2> edges_to_faces;					// for each edge, save adj faces

	for (int i = 0; i < obj_vertices.size(); i++) {
		std::vector<unsigned> empty;
		vertices_to_faces.push_back(empty);
		approx_vert_normals.push_back(glm::vec3(0));
		new_indices.push_back(UINT_MAX);
	}

	// match adjacency info and create new points
	for (int i = 0; i < faces.size(); i++) {
		std::vector<unsigned> cur_face = faces[i];
		int face_size = cur_face.size();

		glm::vec3 cur_normal = glm::vec3(0);
		for (int j = 1; j < face_size - 1; j++) {
			cur_normal += compute_normal(obj_vertices[cur_face[0]], obj_vertices[cur_face[j]], obj_vertices[cur_face[j + 1]]);
		}
		cur_normal = glm::normalize(cur_normal);
		//glm::vec3 cur_normal = compute_normal(obj_vertices[cur_face[0]], obj_vertices[cur_face[1]], obj_vertices[cur_face[2]]);

		// calculate center of the face
		glm::vec4 center = glm::vec4(0);
		for (int j = 0; j < face_size; j++) {
			center += obj_vertices[cur_face[j]];

			// group face for each vertex
			vertices_to_faces[cur_face[j]].push_back(i); 
			approx_vert_normals[cur_face[j]] += cur_normal;
		}
		center /= float(face_size);

		// group face for each edge
		for (int j = 0; j < face_size; j++) {
			glm::vec2 edge = make_edge(cur_face, j);
			bool flipped = cur_face[j] < cur_face[(j + 1) % cur_face.size()] ? false : true;

			// use hashmaps to keep track of face-edge relationship
			if (edges_to_index.find(edge) == edges_to_index.end()) {
				// first occurrence of edge, save edge index
				edges.push_back(edge);
				edges_to_index[edge] = edges.size() - 1;
				edges_to_faces.push_back(glm::uvec2(i, UINT_MAX));
			}
			else if (edges_to_index[edge] != UINT_MAX) {
				// found the second face adjacent to this edge

				// save the other face that the edge is adjacent to
				edges_to_faces[edges_to_index[edge]][1] = i;
				edges_to_index[edge] = UINT_MAX;
			}
			else {
				// > 2 faces adjacent to this edge
				// should not reach this line
				std::cout << "error - edge " << edge << " adjacent to > 2 faces" << std::endl;
			}
		}
		

		std::vector<unsigned> inner_points;

		// construct the inner face points and put together face polys
		for (int j = 0; j < face_size; j++) {
			glm::uvec2 edge1 = make_edge(cur_face, j + face_size - 1);
			glm::uvec2 edge2 = make_edge(cur_face, j);
			glm::vec4 mid1 = (obj_vertices[edge1[0]] + obj_vertices[edge1[1]]) / 2.0f;
			glm::vec4 mid2 = (obj_vertices[edge2[0]] + obj_vertices[edge2[1]]) / 2.0f;

			// face polys - same order as normally
			glm::vec4 new_point = (mid1 + mid2 + center + obj_vertices[cur_face[j]]) / 4.0f;
			new_face_points.push_back(new_point);
			inner_points.push_back(new_face_points.size() - 1);
		}
		new_faces.push_back(inner_points);
	}

	// construct the edge polys
	for (int i = 0; i < edges_to_faces.size(); i ++) {
		glm::uvec2 adj_faces = edges_to_faces[i];
		glm::uvec2 edge = edges[i];
		std::vector<unsigned> new_face;

		// use same order as endpoints appear in the faces
		// (find location of endpoints in original face)
		std::vector<unsigned> face1 = faces[adj_faces[0]];
		std::vector<unsigned> new_face1 = new_faces[adj_faces[0]];
		glm::uvec2 edge1_loc = find_edge(face1, edge);
		if (edge1_loc[0] == UINT_MAX || edge1_loc[1] == UINT_MAX) {
			std::cout << "edge not part of face?" << edge << std::endl;
		}

		bool flipped1 = false;
		if ((edge1_loc[0] + 1) % face1.size() == edge1_loc[1]) {
			// same order
			new_face.push_back(new_face1[edge1_loc[1]]);
			new_face.push_back(new_face1[edge1_loc[0]]);
		}
		else {
			// flipped order
			flipped1 = true;
			new_face.push_back(new_face1[edge1_loc[0]]);
			new_face.push_back(new_face1[edge1_loc[1]]);
		}

		if (adj_faces[1] == UINT_MAX) {
			// just use the other edge
			
			int ind1 = 0, ind2 = 1;
			if (flipped1) {
				ind1 = 1, ind2 = 0;
			}
			int vert1 = face1[edge1_loc[ind1]];
			int vert2 = face1[edge1_loc[ind2]];

			if (new_indices[vert1] == UINT_MAX) {
				new_face_points.push_back(obj_vertices[vert1]);
				new_indices[vert1] = new_face_points.size() - 1;
			}
			new_face.push_back(new_indices[vert1]);

			if (new_indices[vert2] == UINT_MAX) {
				new_face_points.push_back(obj_vertices[vert2]);
				new_indices[vert2] = new_face_points.size() - 1;
			}
			new_face.push_back(new_indices[vert2]);
			
			std::cout << "only 1 face for " << edge << std::endl;
			continue;
		}
		else {
			std::vector<unsigned> face2 = faces[adj_faces[1]];
			std::vector<unsigned> new_face2 = new_faces[adj_faces[1]];
			glm::uvec2 edge2_loc = find_edge(face2, edge);
			if (edge2_loc[0] == UINT_MAX || edge2_loc[1] == UINT_MAX) {
				std::cout << "edge not part of face? " << edge << std::endl;
			}

			if ((edge2_loc[0] + 1) % face2.size() == edge2_loc[1]) {
				// same order
				new_face.push_back(new_face2[edge2_loc[1]]);
				new_face.push_back(new_face2[edge2_loc[0]]);
			}
			else {
				// flipped order
				new_face.push_back(new_face2[edge2_loc[0]]);
				new_face.push_back(new_face2[edge2_loc[1]]);
			}
		}
		new_faces.push_back(new_face);
	}
	
	// vertex polys - use angle sizes to figure out radial order
	// STILL SOMETHING WRONG WITH RADIAL ORDER
	for (int i = 0; i < obj_vertices.size(); i++) {
		std::vector<unsigned> adj_faces = vertices_to_faces[i];
		if (adj_faces.size() >= 3) {
			glm::vec3 normal = glm::normalize(approx_vert_normals[i]);
			glm::vec4 cur_vert = obj_vertices[i];

			std::vector<unsigned> new_face;
			std::vector<float> angles;

			glm::vec4 center = glm::vec4(0);
			for (int j = 0; j < adj_faces.size(); j++) {
				int face_ind = adj_faces[j];
				unsigned ind = find_vertex(faces[face_ind], i);

				unsigned vert_ind = new_faces[face_ind][ind]; // corresponding new vert
				center += new_face_points[vert_ind];
			}
			center /= adj_faces.size();

			// pick a neighboring vertex as an anchor
			int face_anchor_ind = adj_faces[0];
			unsigned ind = find_vertex(faces[face_anchor_ind], i); // vertex's position in the face
			unsigned anchor_ind = new_faces[face_anchor_ind][ind]; // get corresponding new face point
			new_face.push_back(anchor_ind);
			angles.push_back(0);

			// sort other vertices by angle formed
			for (int j = 1; j < adj_faces.size(); j++) {
				int face_ind = adj_faces[j];
				unsigned ind = find_vertex(faces[face_ind], i);
			
				unsigned vert_ind = new_faces[face_ind][ind]; // corresponding new vert
				float angle = compute_angle(
					new_face_points[anchor_ind], 
					center,
					new_face_points[vert_ind], 
					normal);
				
				if (angle >= angles[angles.size() - 1]) {
					angles.push_back(angle);
					new_face.push_back(vert_ind);
				}
				else {
					bool inserted = false;
					for (int k = 0; k < angles.size() && !inserted; k++) {
						if (angle <= angles[k]) {
							angles.insert(angles.begin() + k, angle);
							new_face.insert(new_face.begin() + k, vert_ind);
							
							inserted = true;
						}
					}
					if (!inserted) {
						std::cout << angle << std::endl;
						std::cout << angles[angles.size() - 1] << std::endl;
					}
				}
			}

			if (new_face.size() != adj_faces.size()) {
				std::cout << "ERROR " << new_face.size() << " " << adj_faces.size() << std::endl;
			}

			if (new_face.size() >= 3) {
				new_faces.push_back(new_face);
			}
			else {
				std::cout << "error - created degen face of size " << new_face.size() << std::endl;
			}
		}
	}

	faces = new_faces;
	obj_vertices = new_face_points;
}

// converts a given face to tris, assuming face is defined counterclockwise
std::vector<std::vector<unsigned>> Mesh::convert_tris(std::vector<unsigned int> face) const {
	std::vector<std::vector<unsigned>> result;
	int num_verts = face.size();

	if (num_verts < 3) {
		std::cout << "error - degenerate face with " << face.size() << " verts" << std::endl;
	}
	else if (num_verts == 3) {
		result.push_back(face);
	}
	else {
		// divide larger polys into tris
		for (int i = 1; i < num_verts - 1; i++) {
			std::vector<unsigned> tri1 = { face[0], face[i], face[i + 1] };
			result.push_back(tri1);
		}
	}

	return result;
}

// convert all face vectors into uvec3s (tris) for rendering
void Mesh::get_tris(std::vector<glm::uvec3>& tris) const {
	tris.clear();

	for(int i = 0; i < faces.size(); i ++) {
		std::vector<std::vector<unsigned>> split_tris = convert_tris(faces[i]);
		for (int j = 0; j < split_tris.size(); j ++) {
			std::vector<unsigned> cur_tri = split_tris[j];
			if (cur_tri.size() == 3) {
				tris.push_back(glm::uvec3(cur_tri[0], cur_tri[1], cur_tri[2]));
			}
			else {
				std::cout << "error - encountered face with size = " << cur_tri.size() << std::endl;
			}
		}
	}
}

// helper to make sure edge[0] <= edge[1], enforce consistency in hash map
glm::uvec2 make_edge(std::vector<unsigned> cur_face, unsigned j) {
	int verts_per_face = cur_face.size();

	glm::uvec2 edge = glm::uvec2(cur_face[j % verts_per_face], cur_face[(j + 1) % verts_per_face]);
	if (edge[0] > edge[1]) {
		edge = glm::uvec2(edge[1], edge[0]);
	}

	return edge;
}

// locates the edge in the face
glm::uvec2 find_edge(std::vector<unsigned> cur_face, glm::uvec2 edge) {
	glm::uvec2 result = glm::uvec2(UINT_MAX, UINT_MAX);
	for (int j = 0; j < cur_face.size(); j++) {
		if (cur_face[j] == edge[0]) {
			result[0] = j;
		}
		if (cur_face[j] == edge[1]) {
			result[1] = j;
		}
	}
	return result;
}

// locates the vertex in the face
unsigned find_vertex(std::vector<unsigned> cur_face, unsigned vert) {
	unsigned result = UINT_MAX;
	for (int j = 0; j < cur_face.size(); j++) {
		if (cur_face[j] == vert) {
			return j;
		}
	}
	return result;
}

float compute_angle(glm::vec4 anchor, glm::vec4 center, glm::vec4 v, glm::vec3 n) {
	glm::vec4 side = anchor - center;
	glm::vec3 side1 = glm::normalize(glm::vec3(side[0], side[1], side[2]));

	side = v - center;
	glm::vec3 side2 = glm::normalize(glm::vec3(side[0], side[1], side[2]));
	if (glm::length(side1 - side2) > 1.99f) {
		return M_PI;
	}

	float angle = glm::acos(glm::dot(side1, side2));
	glm::vec3 normal = glm::normalize(glm::cross(side1, side2));
	if (glm::dot(normal, n) < 0) {
		angle += M_PI;
	}
	return angle;
}

// calculates a normal for a given face
glm::vec3 compute_normal(glm::vec4 v0, glm::vec4 v1, glm::vec4 v2) {
	glm::vec4 side = v1 - v0;
	glm::vec3 side1 = glm::vec3(side[0], side[1], side[2]);

	side = v2 - v0;
	glm::vec3 side2 = glm::vec3(side[0], side[1], side[2]);
	return glm::normalize(glm::cross(side1, side2));
}