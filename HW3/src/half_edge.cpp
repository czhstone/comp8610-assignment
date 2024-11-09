#include <include/half_edge.hpp>
#include <Eigen/Dense>

/*
#################################################################################
#                       Vertex-related Helper Functions                         #
#################################################################################
*/

// Optinal TODO: Iterate through all neighbour vertices around the vertex
// Helpful when you implement the average degree computation of the mesh
std::vector<std::shared_ptr<Vertex>> Vertex::neighbor_vertices() {
    std::vector<std::shared_ptr<Vertex>> neighborhood;

    std::shared_ptr<HalfEdge> start = this -> he;
    std::shared_ptr<HalfEdge> current = start;
    
    do {

        neighborhood.push_back(current->twin->vertex);

        current = current->twin->next;
        
    } while (current != this->he);  

    return neighborhood; 

    
}


// TODO: Iterate through all half edges pointing away from the vertex
std::vector<std::shared_ptr<HalfEdge>> Vertex::neighbor_half_edges() {
    std::vector<std::shared_ptr<HalfEdge>> neighborhood;

    if (!he) return neighborhood;  // Return empty if no half-edge linked

    std::shared_ptr<HalfEdge> start = he;
    do {
        neighborhood.push_back(he);
        std::shared_ptr<HalfEdge> twin = he->twin;
        he = twin ? twin->next : nullptr;
    } while (he != start && he);

    return neighborhood;

}


// TODO: Computate quadratic error metrics coefficient, which is a 5-d vector associated with each vertex
/*
    HINT:
        Please refer to homework description about how to calculate each element in the vector.
        The final results is stored in class variable "this->qem_coff"
*/
void Vertex::compute_qem_coeff() {
    // Reset QEM coefficients with the appropriate size and initialize to zero
    this->qem_coff = Eigen::VectorXf(5);
    this->qem_coff.setZero();

    // Compute the sum of positions and the sum of squared positions of all neighbor vertices
    Eigen::Vector3f totalNeighborPositions = Eigen::Vector3f::Zero();
    float totalSquaredNorms = 0.0f;
    std::vector<std::shared_ptr<Vertex>> neighbors = this->neighbor_vertices();
    size_t numberOfNeighbors = neighbors.size();

    for (const auto& neighbor : neighbors) {
        totalNeighborPositions += neighbor->pos;  // Accumulate positions
        totalSquaredNorms += neighbor->pos.squaredNorm();  // Accumulate squared norms
    }

    // Fill in the Quadratic Error Metrics (QEM) coefficients
    this->qem_coff[0] = static_cast<float>(numberOfNeighbors); // The count of neighbors
    this->qem_coff.segment<3>(1) = totalNeighborPositions;     // Sum of neighbor positions
    this->qem_coff[4] = totalSquaredNorms;                     // Sum of squared positions

}


/*
#################################################################################
#                         Face-related Helper Functions                         #
#################################################################################
*/

// TODO: Iterate through all member vertices of the face
std::vector<std::shared_ptr<Vertex>> Face::vertices() {
    std::vector<std::shared_ptr<Vertex>> member_vertices;

    if (!he) return member_vertices;  // Return empty if no half-edge linked

    std::shared_ptr<HalfEdge> start = he;
    do {
        member_vertices.push_back(he->vertex);
        he = he->next;
    } while (he != start && he);


    return member_vertices;
}


// TODO: implement this function to compute the area of the triangular face
float Face::get_area(){
    float area;

    if (!he) return 0.0;  // Return 0 if no half-edge linked

    Eigen::Vector3f A = he->vertex->pos;
    Eigen::Vector3f B = he->next->vertex->pos;
    Eigen::Vector3f C = he->next->next->vertex->pos;

    return 0.5 * ((B - A).cross(C - A)).norm();

}

// TODO: implement this function to compute the signed volume of the triangular face
// reference: http://chenlab.ece.cornell.edu/Publication/Cha/icip01_Cha.pdf eq.(5)
float Face::get_signed_volume(){
    float volume;
    if (!he) return 0.0;  // Return 0 if no half-edge linked

    Eigen::Vector3f A = he->vertex->pos;
    Eigen::Vector3f B = he->next->vertex->pos;
    Eigen::Vector3f C = he->next->next->vertex->pos;

    return (1.0 / 6.0) * A.dot(B.cross(C));

}


/*
#################################################################################
#                         Edge-related Helper Functions                         #
#################################################################################
*/

/*
    TODO: 
        Compute the contraction information for the edge (v1, v2), which will be used later to perform edge collapse
            (i) The optimal contraction target v*
            (ii) The quadratic error metrics QEM, which will become the cost of contracting this edge
        The final results is stored in class variable "this->verts_contract_pos" and "this->qem"
    Please refer to homework description for more details
*/
void Edge::compute_contraction() {
    // Retrieve the vertices of the edge
    auto v1 = this->he->vertex;
    auto v2 = this->he->twin->vertex;

    // Calculate the combined Quadratic Error Metric (QEM) coefficients
    Eigen::VectorXf combinedQEMCoefficients = v1->qem_coff + v2->qem_coff;

    // Compute the optimal position for the contracted vertex
    Eigen::Vector3f optimalPosition = combinedQEMCoefficients.segment<3>(1) / combinedQEMCoefficients[0];
    this->verts_contract_pos = optimalPosition;

    // Compute the quadratic error at the optimal position
    // E(v) = a * ||v||^2 - 2 * b^T * v + c （simplified from the original equation）
    float a = combinedQEMCoefficients[0];
    Eigen::Vector3f b = combinedQEMCoefficients.segment<3>(1);
    float c = combinedQEMCoefficients[4];

    float vTv = optimalPosition.transpose().dot(optimalPosition);
    float bv = b.transpose().dot(optimalPosition);

    this->qem = a * vTv - 2 * bv + c;
}


/*
    TODO: 
        Perform edge contraction functionality, which we write as (v1 ,v2) → v*, 
            (i) Moves the vertex v1 to the new position v*, remember to update all corresponding attributes,
            (ii) Connects all incident edges of v1 and v2 to v*, and remove the vertex v2,
            (iii) All faces, half edges, and edges associated with this collapse edge will be removed.
    HINT: 
        (i) Pointer reassignments
        (ii) When you want to remove mesh components, simply set their "exists" attribute to False
    Please refer to homework description for more details
*/
void Edge::edge_contraction() {
    // Retrieve the two vertices of the edge
    auto v1 = this->he->vertex;
    auto v2 = this->he->twin->vertex;

    // Update v1's position and QEM coefficients
    v1->pos = this->verts_contract_pos;
    v1->qem_coff = v1->qem_coff + v2->qem_coff;

    // Redirect all adjacent half-edges from v2 to v1
    std::vector<std::shared_ptr<HalfEdge>> v2Edges = v2->neighbor_half_edges();
    for (auto& he : v2Edges) {
        if (he->exists) {
            he->vertex = v1;
        }
    }

    // Mark half-edges and edges associated with the edge as non-existent
    auto he1 = this->he;
    auto he2 = this->he->twin;
    auto he1Next = he1->next;
    auto he1NextNext = he1Next->next;
    auto he2Next = he2->next;
    auto he2NextNext = he2Next->next;

    he1Next->exists = false;
    he1Next->edge->exists = false;
    he1NextNext->exists = false;

    he2Next->exists = false;
    he2NextNext->exists = false;
    he2NextNext->edge->exists = false;

    // Delete the original edge and the corresponding vertex v2
    he1->exists = false;
    he2->exists = false;
    he1->face->exists = false;
    he2->face->exists = false;
    this->exists = false;
    v2->exists = false;

    // Reset twin references for opposing half-edges
    he1NextNext->twin->twin = he1Next->twin;
    he1Next->twin->twin = he1NextNext->twin;

    he2NextNext->twin->twin = he2Next->twin;
    he2Next->twin->twin = he2NextNext->twin;

    // Reset references for related half-edges and edges
    he1NextNext->edge->he = he1NextNext->twin;
    he2Next->edge->he = he2Next->twin;

    he1Next->twin->edge = he1NextNext->edge;
    he2NextNext->twin->edge = he2Next->edge;

    // Update v1's half-edge reference to a valid half-edge
    v1->he = he1NextNext->twin;

    // Update half-edge references for vertices A and B

    he1Next->twin->vertex->he = he1Next->twin;
    he2Next->twin->vertex->he = he2Next->twin;

}