#include <glm/glm.hpp>
#include <vector>
#include <string>


class Mesh {
public:
	Mesh();			// default constructor - initialize a unit cube
	Mesh(char *fn);	// loads data from an obj file, passed on command line

	~Mesh();

	bool is_dirty() const;
	void set_clean();

	std::vector<glm::vec4> obj_vertices;

	// change the subdivision scheme
	void cycle_scheme();

	// perform 1 iteration of the correct subdivide algorithm
	void subdivide();

	// returns faces as tris for rendering
	void get_tris(std::vector<glm::uvec3>& tris) const;

	// re-initialize to initial state
	void initialize();

private:
	bool dirty_ = false;
	char* filename_ = nullptr;

	enum class scheme { CATMULL, LOOP, DOO_SABIN, END };
	Mesh::scheme subdivision_type;
	std::string scheme_name();

	void initialize_cube();
	void load_from_file();

	void catmull_clark();
	void doo_sabin();
	void loop();

	std::vector<std::vector<unsigned>> convert_tris(std::vector<unsigned int> face) const;

	std::vector<std::vector<unsigned int>> faces;
};