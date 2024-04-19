/* ----------------------------------------------------------------------------
 * This problem solves the incompresible steady-state Navier-Stokes equations
 * using a straightforward approach.
 * ----------------------------------------------------------------------------
 */

#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

using namespace dealii;

// This structure is used to create a small parameter file in deal.ii.
// In practice, it is useful to have a parameter file in order to run
// several simulations without compiling the program everytime we want
// to change something. In this case, it consists of only three parameters.
struct Settings
{
  bool
  try_parse(const std::string &prm_filename);

  enum ProblemType
  {
    couette,
    poiseuille,
    manufactured,
    flow_around_cylinder
  };

  ProblemType problem_type;
  double      viscosity;
  int number_of_initial_refinement;
};

bool
Settings::try_parse(const std::string &prm_filename)
{
  ParameterHandler prm;
  prm.declare_entry(
    "problem type",
    "couette",
    Patterns::Selection("couette | poiseuille | manufactured |flow around cylinder"),
    "Problem type <couette | poiseuille | manufactured |flow around cylinder>");
  prm.declare_entry("viscosity", "1", Patterns::Double(), "Viscosity");
  prm.declare_entry("number_of_initial_refinement", "1", Patterns::Double(), "number_of_initial_refinement");

  if (prm_filename.size() == 0)
    {
      std::cout
        << "****  Error: No input file provided!\n"
        << "****  Error: Call this program as './matrix_based_non_linear_poisson input.prm\n"
        << '\n'
        << "****  You may want to use one of the input files in this\n"
        << "****  directory, or use the following default values\n"
        << "****  to create an input file:\n";
      prm.print_parameters(std::cout, ParameterHandler::Text);
      return false;
    }

  try
    {
      prm.parse_input(prm_filename);
    }
  catch (std::exception &e)
    {
      std::cerr << e.what() << std::endl;
      return false;
    }

  if (prm.get("problem type") == "couette")
    this->problem_type = couette;
  else if (prm.get("problem type") == "poiseuille")
    this->problem_type = poiseuille;
  else if (prm.get("problem type") == "flow around cylinder")
    this->problem_type = flow_around_cylinder;
  else if (prm.get("problem type") == "manufactured")
    this->problem_type = manufactured;
  else
    AssertThrow(false, ExcNotImplemented());

  this->viscosity = prm.get_double("viscosity");
  this->number_of_initial_refinement = prm.get_double("number_of_initial_refinement");

  return true;
}

// This function is used when non-homogeneous dirichlet boundary conditions
// are needed for the velocity.
template <int dim>
class VelocityBoundaryFunction : public Function<dim>
{
public:
  VelocityBoundaryFunction()
    : Function<dim>(dim + 1)
  {}

  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override;
};

// TODO: complete the value function that uses the component parameter
// to return the appropriate value of each component of the velocity.
template <int dim>
double
VelocityBoundaryFunction<dim>::value(const Point<dim> & p,
                                     const unsigned int component) const
{
  double epsilonMMS = 0.001;
  if(component==0 ){
    return (sin(p[0]*p[0] + p[1]*p[1]) + epsilonMMS);
  }
  else if(component==1 ){
    return (cos(p[0]*p[0] + p[1]*p[1]) + epsilonMMS);
  }
}

template <int dim>
class PressureBoundaryFunction : public Function<dim>
{
public:
  PressureBoundaryFunction()
    : Function<dim>(dim + 1)
  {}

  virtual double
  value(const Point<dim> &p, const unsigned int component = 0) const override;
};

// TODO: complete the value function that uses the component parameter
// to return the appropriate value of each component of the velocity.
template <int dim>
double
PressureBoundaryFunction<dim>::value(const Point<dim> & p,
                                     const unsigned int component) const
{
    return (sin(p[0]*p[0] + p[1]*p[1]) + 2.);

}

// Main class to solve the steady=state Navier-Stokes equations
template <int dim>
class SteadyStateNavierStokes
{
public:
  SteadyStateNavierStokes(const Settings &parameters);

  /**
   * @brief run This is the main function fo the class, where we set up the triangulation, the system,
   * and have the newton solver.
   * TODO: go and write the newton solver. 
   */
  void
  run();

private:
  /**
   * @brief setup_triangulation Sets-up the triangulations. This is where you will create the different grids.
   * TODO: go and create the triangulation for each problem.
   */
  void
  setup_triangulation();

  /**
   * @brief setup_system This part generates the sparsity pattern and allocates the memory for the matrix,
   * the right-hand side and the solution of both systems that need to be
   * solved.
   * TODO: go and complete the boundary conditions for each problem.
   */
  void
  setup_system();

  void
  errorL2();

  /**
   * @brief assemble_system Assembles the matrix and the right-hand side of the momentum equation
   * TODO: go and complete the assembly
   */
  void
  assemble_system();

  /**
   * @brief solve Solves the momentum to have a velocity that does not satisfy the continuity eqn
   */
  void
  solve_linear_system();

  /**
   * @brief output_results Outputs the results of the simulation into vtk files for Paraview.
   */
  void
  output_results(unsigned int iter) const;

  Triangulation<dim> triangulation;

  FESystem<dim>             fe;
  DoFHandler<dim>           dof_handler;
  AffineConstraints<double> constraints;
  AffineConstraints<double> zero_constraints;
  SparsityPattern           sparsity_pattern;
  SparseMatrix<double>      system_matrix;
  Vector<double>            system_rhs;

  Vector<double> solution;
  Vector<double> delta_solution;

  Settings    parameters;
  double      viscosity;
  int number_of_initial_refinement;
  std::string test_case;
};

template <int dim>
SteadyStateNavierStokes<dim>::SteadyStateNavierStokes(
  const Settings &parameters)
  : fe(FE_Q<dim>(2), dim, FE_Q<dim>(1), 1)
  , dof_handler(triangulation)
  , parameters(parameters)
{
  if (parameters.problem_type == Settings::couette)
    {
      test_case = "couette";
    }
  else if (parameters.problem_type == Settings::poiseuille)
    {
      test_case = "poiseuille";
    }
  else if (parameters.problem_type == Settings::manufactured)
    {
      test_case = "manufactured";
    }
  else if (parameters.problem_type == Settings::flow_around_cylinder)
    {
      test_case = "flow_around_cylinder";
    }

  viscosity = parameters.viscosity;
  number_of_initial_refinement = parameters.number_of_initial_refinement;
}

template <int dim>
void
SteadyStateNavierStokes<dim>::setup_triangulation()
{
  if (parameters.problem_type == Settings::couette)
    {
      // TODO: create the triangulation for the couette case.
      // Remember to refine the triangulation globally.
      int number_of_initial_refinement = 5;
      Point<dim> p1(0,0);
      Point<dim> p2(8,4);
      GridGenerator::hyper_rectangle(triangulation, p1, p2, true );
      triangulation.refine_global(number_of_initial_refinement);
    }

  if (parameters.problem_type == Settings::poiseuille)
    {
      // TODO: create the triangulation for the poiseuille case.
      // Remember to refine the triangulation globally.
      int number_of_initial_refinement = 5;
      Point<dim> p1(0,0);
      Point<dim> p2(8,4);
      GridGenerator::hyper_rectangle(triangulation, p1, p2, true );
      triangulation.refine_global(number_of_initial_refinement);
    }
  else if (parameters.problem_type == Settings::manufactured)
    {
      std::cout<<"manufactured triangulation CYL..."<<std::endl;
       
       GridGenerator::channel_with_cylinder(triangulation, 0.03, 2, 2.0, true );
       triangulation.refine_global(number_of_initial_refinement);
    }
  else if (parameters.problem_type == Settings::flow_around_cylinder)
    {
      // TODO: create the triangulation for the flow around a cylinder.
      // Use the GridGenerator of deal.ii and remember to refine the
      // triangulation globally.
       int number_of_initial_refinement = 4;
       GridGenerator::channel_with_cylinder(triangulation, 0.03, 2, 2.0, true );
       triangulation.refine_global(number_of_initial_refinement);
    }

  std::cout << "---------------------------------------------------------"
            << std::endl;
  std::cout << "------------------- MESH INFORMATION --------------------"
            << std::endl;
  std::cout << "---------------------------------------------------------"
            << std::endl;
  std::cout << "   Number of active cells: " << triangulation.n_active_cells()
            << std::endl
            << "   Total number of cells: " << triangulation.n_cells()
            << std::endl;
}

template <int dim>
void
SteadyStateNavierStokes<dim>::setup_system()
{
  std::cout << "    Setup system ..." << std::endl;

  // Distribute DoFS
  dof_handler.distribute_dofs(fe);

  std::cout << "   Total number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;

  // Create sparsity pattern
  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);
  system_matrix.reinit(sparsity_pattern);

  // Initialize vectors for each system
  delta_solution.reinit(dof_handler.n_dofs());
  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());

  FEValuesExtractors::Vector velocities(0);
  FEValuesExtractors::Scalar pressure(dim);

  // We will have two constraints object, one for the initial 
  // newton iteration (constraints) and one for the update in
  // the following iterations (zero constraints). Fill them 
  // for each case

  // Fill constraints for each test case
  constraints.clear();

  if (parameters.problem_type == Settings::couette)
    {
      // TODO: implement the initial conditions for this case
      VectorTools::interpolate_boundary_values(dof_handler,
                                               2,
                                               Functions::ZeroFunction<dim>(dim + 1),
                                               constraints,
                                               fe.component_mask(velocities));
      VectorTools::interpolate_boundary_values(dof_handler,
                                               3,
                                               VelocityBoundaryFunction<dim>(), 
                                               constraints,
                                               fe.component_mask(velocities));
      VectorTools::interpolate_boundary_values(dof_handler,
                                               1,
                                               Functions::ZeroFunction<dim>(dim + 1), 
                                               constraints,
                                               fe.component_mask(pressure));

    }
  else if (parameters.problem_type == Settings::manufactured)
    {
      // Velocity:
      VectorTools::interpolate_boundary_values(dof_handler,
                                               0,
                                               VelocityBoundaryFunction<dim>(), 
                                               constraints,
                                               fe.component_mask(velocities));
      VectorTools::interpolate_boundary_values(dof_handler,
                                               1,
                                               VelocityBoundaryFunction<dim>(), 
                                               constraints,
                                               fe.component_mask(velocities));                                         
      VectorTools::interpolate_boundary_values(dof_handler,
                                               2,
                                               VelocityBoundaryFunction<dim>(),
                                               constraints,
                                               fe.component_mask(velocities));
      VectorTools::interpolate_boundary_values(dof_handler,
                                               3,
                                               VelocityBoundaryFunction<dim>(), 
                                               constraints,
                                               fe.component_mask(velocities));
      // Pressure
      VectorTools::interpolate_boundary_values(dof_handler,
                                               0,
                                               PressureBoundaryFunction<dim>(), 
                                               constraints,
                                               fe.component_mask(pressure));
      VectorTools::interpolate_boundary_values(dof_handler,
                                               1,
                                               PressureBoundaryFunction<dim>(), 
                                               constraints,
                                               fe.component_mask(pressure));
      VectorTools::interpolate_boundary_values(dof_handler,
                                               2,
                                               PressureBoundaryFunction<dim>(), 
                                               constraints,
                                               fe.component_mask(pressure));                                         
      VectorTools::interpolate_boundary_values(dof_handler,
                                               3,
                                               PressureBoundaryFunction<dim>(), 
                                               constraints,
                                               fe.component_mask(pressure));
    }
  else if (parameters.problem_type == Settings::poiseuille)
    {
      // TODO: implement the boundary conditions for this case
      VectorTools::interpolate_boundary_values(dof_handler,
                                               0,
                                               VelocityBoundaryFunction<dim>(), 
                                               constraints,
                                               fe.component_mask(velocities));
      VectorTools::interpolate_boundary_values(dof_handler,
                                               2,
                                               Functions::ZeroFunction<dim>(dim + 1),
                                               constraints,
                                               fe.component_mask(velocities));
      VectorTools::interpolate_boundary_values(dof_handler,
                                               3,
                                               Functions::ZeroFunction<dim>(dim + 1), 
                                               constraints,
                                               fe.component_mask(velocities));
      VectorTools::interpolate_boundary_values(dof_handler,
                                               1,
                                               Functions::ZeroFunction<dim>(dim + 1), 
                                               constraints,
                                               fe.component_mask(pressure));
    }
  else if (parameters.problem_type == Settings::flow_around_cylinder)
    {
      // TODO: implement the boundary conditions for this case
      VectorTools::interpolate_boundary_values(dof_handler,
                                               0,
                                               VelocityBoundaryFunction<dim>(), 
                                               constraints,
                                               fe.component_mask(velocities));
      VectorTools::interpolate_boundary_values(dof_handler,
                                               2,
                                               Functions::ZeroFunction<dim>(dim + 1),
                                               constraints,
                                               fe.component_mask(velocities));
      VectorTools::interpolate_boundary_values(dof_handler,
                                               3,
                                               VelocityBoundaryFunction<dim>(), 
                                               constraints,
                                               fe.component_mask(velocities));
      VectorTools::interpolate_boundary_values(dof_handler,
                                               1,
                                               Functions::ZeroFunction<dim>(dim + 1), 
                                               constraints,
                                               fe.component_mask(pressure));
    }

  constraints.close();

  // Fill zero constraints for each test case
  zero_constraints.clear();

  if (parameters.problem_type == Settings::couette)
    {
      // TODO: implement the boundary conditions for this case

      VectorTools::interpolate_boundary_values(dof_handler,
                                               2,
                                               Functions::ZeroFunction<dim>(dim + 1),
                                               zero_constraints,
                                               fe.component_mask(velocities));
      VectorTools::interpolate_boundary_values(dof_handler,
                                               3,
                                               Functions::ZeroFunction<dim>(dim + 1), 
                                               zero_constraints,
                                               fe.component_mask(velocities));
      VectorTools::interpolate_boundary_values(dof_handler,
                                               1,
                                               Functions::ZeroFunction<dim>(dim + 1),
                                               zero_constraints,
                                               fe.component_mask(pressure));
    }

  else if (parameters.problem_type == Settings::manufactured)
    {
      // Velocity
      VectorTools::interpolate_boundary_values(dof_handler,
                                               0,
                                               Functions::ZeroFunction<dim>(dim + 1),
                                               zero_constraints,
                                               fe.component_mask(velocities));
      VectorTools::interpolate_boundary_values(dof_handler,
                                               1,
                                               Functions::ZeroFunction<dim>(dim + 1),
                                               zero_constraints,
                                               fe.component_mask(velocities));
      VectorTools::interpolate_boundary_values(dof_handler,
                                               2,
                                               Functions::ZeroFunction<dim>(dim + 1),
                                               zero_constraints,
                                               fe.component_mask(velocities));
      VectorTools::interpolate_boundary_values(dof_handler,
                                               3,
                                               Functions::ZeroFunction<dim>(dim + 1), 
                                               zero_constraints,
                                               fe.component_mask(velocities));
      // Pressure                                         
      VectorTools::interpolate_boundary_values(dof_handler,
                                               0,
                                               Functions::ZeroFunction<dim>(dim + 1),
                                               zero_constraints,
                                               fe.component_mask(pressure));
      VectorTools::interpolate_boundary_values(dof_handler,
                                               1,
                                               Functions::ZeroFunction<dim>(dim + 1),
                                               zero_constraints,
                                               fe.component_mask(pressure));
      VectorTools::interpolate_boundary_values(dof_handler,
                                               2,
                                               Functions::ZeroFunction<dim>(dim + 1),
                                               zero_constraints,
                                               fe.component_mask(pressure));
      VectorTools::interpolate_boundary_values(dof_handler,
                                               3,
                                               Functions::ZeroFunction<dim>(dim + 1),
                                               zero_constraints,
                                               fe.component_mask(pressure));
    }
  else if (parameters.problem_type == Settings::poiseuille)
    {
      // TODO: implement the boundary conditions for this case
      VectorTools::interpolate_boundary_values(dof_handler,
                                               0,
                                               Functions::ZeroFunction<dim>(dim + 1),
                                               zero_constraints,
                                               fe.component_mask(velocities));
      VectorTools::interpolate_boundary_values(dof_handler,
                                               2,
                                               Functions::ZeroFunction<dim>(dim + 1),
                                               zero_constraints,
                                               fe.component_mask(velocities));
      VectorTools::interpolate_boundary_values(dof_handler,
                                               3,
                                               Functions::ZeroFunction<dim>(dim + 1), 
                                               zero_constraints,
                                               fe.component_mask(velocities));
      VectorTools::interpolate_boundary_values(dof_handler,
                                               1,
                                               Functions::ZeroFunction<dim>(dim + 1),
                                               zero_constraints,
                                               fe.component_mask(pressure));
    }
  else if (parameters.problem_type == Settings::flow_around_cylinder)
    {
      // TODO: implement the boundary conditions for this case
      VectorTools::interpolate_boundary_values(dof_handler,
                                               0,
                                               Functions::ZeroFunction<dim>(dim + 1),
                                               zero_constraints,
                                               fe.component_mask(velocities));
      VectorTools::interpolate_boundary_values(dof_handler,
                                               2,
                                               Functions::ZeroFunction<dim>(dim + 1),
                                               zero_constraints,
                                               fe.component_mask(velocities));
      VectorTools::interpolate_boundary_values(dof_handler,
                                               3,
                                               Functions::ZeroFunction<dim>(dim + 1), 
                                               zero_constraints,
                                               fe.component_mask(velocities));
      VectorTools::interpolate_boundary_values(dof_handler,
                                               1,
                                               Functions::ZeroFunction<dim>(dim + 1),
                                               zero_constraints,
                                               fe.component_mask(pressure));
    }

  zero_constraints.close();

  // Apply boundary conditions to solution vector for first iteration
  constraints.distribute(solution);
}

template <int dim>
void
SteadyStateNavierStokes<dim>::assemble_system()
{
  std::cout << "    Assembling system ..." << std::endl;

  system_matrix = 0;
  system_rhs    = 0;

  QGauss<dim> quadrature_formula(fe.degree + 1);

  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  const unsigned int n_q_points = fe_values.n_quadrature_points;

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  // Masks to distinguish between velocity and pressure
  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar pressure(dim);

  std::vector<Tensor<1, dim>> previous_velocity(n_q_points);
  std::vector<double>         previous_velocity_divergence(n_q_points);
  std::vector<Tensor<2, dim>> previous_velocity_gradient(n_q_points);

  std::vector<double> previous_pressure(n_q_points);

  std::vector<double>         div_phi_u(dofs_per_cell);
  std::vector<Tensor<1, dim>> phi_u(dofs_per_cell);
  std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
  std::vector<double>         phi_p(dofs_per_cell);

  double epsilonMMS = 0.001;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);

      cell_matrix = 0;
      cell_rhs    = 0;

      fe_values[velocities].get_function_values(solution, previous_velocity);
      fe_values[velocities].get_function_gradients(solution,
                                                   previous_velocity_gradient);

      for (unsigned int q = 0; q < n_q_points; q++)
        {
          previous_velocity_divergence[q] =
            trace(previous_velocity_gradient[q]);
        }

      fe_values[pressure].get_function_values(solution, previous_pressure);

      for (unsigned int q = 0; q < n_q_points; q++)
        {
          // TODO: get velocity and pressure shape functions
          for (unsigned int k = 0; k < dofs_per_cell; ++k)
            {
              div_phi_u[k]  = fe_values[velocities].divergence(k, q);
              grad_phi_u[k] = fe_values[velocities].gradient(k, q);
              phi_u[k]      = fe_values[velocities].value(k, q);
              phi_p[k]      = fe_values[pressure].value(k, q);
            }
          for (const unsigned int i : fe_values.dof_indices())
            {

              for (const unsigned int j : fe_values.dof_indices())
                {
                  
                  cell_matrix(i, j) += 
                    (viscosity * scalar_product(grad_phi_u[i], grad_phi_u[j]) +
                     phi_u[i] * (previous_velocity_gradient[q] * phi_u[j]) +
                     phi_u[i] * (grad_phi_u[j] * previous_velocity[q]) 
                     - div_phi_u[i] * phi_p[j]
                     - phi_p[i] * div_phi_u[j] 
                     ) *
                    fe_values.JxW(q);           
                }
              
              double x_q = fe_values.get_quadrature_points()[q][0];
              double x_q2 = x_q*x_q;
              double y_q = fe_values.get_quadrature_points()[q][1];
              double y_q2 = y_q*y_q;
              double sin2 = std::sin(x_q2 + y_q2);
              double cos2 = std::cos(x_q2 + y_q2);
              

              // Source term continuity
              double Sp;
              Sp = 2.*x_q*cos2 - 2.*y_q*sin2;
              
              // Source term momentum
              Tensor<1, dim> Su; 
              Su[0] = 4.*viscosity*(x_q2*sin2 + y_q2*sin2 - cos2) 
                    + 2.*x_q*(epsilonMMS + sin2)*cos2 + 2.*x_q*cos2 
                    + 2.*y_q*(epsilonMMS + cos2)*cos2;

              Su[1] = 4.*viscosity*(x_q2*cos2 + y_q2*cos2 + sin2) 
                    - 2.*x_q*(epsilonMMS + sin2)*sin2 
                    - 2.*y_q*(epsilonMMS + cos2)*sin2 + 2.*y_q*cos2;

              cell_rhs(i) += 
                (- viscosity * scalar_product(grad_phi_u[i], previous_velocity_gradient[q]) 
                 - phi_u[i] * (previous_velocity_gradient[q] * previous_velocity[q] - Su) 
                 + div_phi_u[i] * previous_pressure[q] 
                 + phi_p[i] * (previous_velocity_divergence[q] - Sp) 
                 ) * fe_values.JxW(q);
            }
        }

      cell->get_dof_indices(local_dof_indices);
      zero_constraints.distribute_local_to_global(
        cell_matrix, cell_rhs, local_dof_indices, system_matrix, system_rhs);
    }
}

template <int dim>
void
SteadyStateNavierStokes<dim>::errorL2()
{
  std::cout << "    L2 error ..." << std::endl;


  QGauss<dim> quadrature_formula(fe.degree + 1);

  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                            update_quadrature_points | update_JxW_values);

  const unsigned int n_q_points = fe_values.n_quadrature_points;

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  // Masks to distinguish between velocity and pressure
  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar pressure(dim);

  std::vector<Tensor<1, dim>> previous_velocity(n_q_points);
  std::vector<double>         previous_velocity_divergence(n_q_points);
  std::vector<Tensor<2, dim>> previous_velocity_gradient(n_q_points);

  std::vector<double> previous_pressure(n_q_points);

  std::vector<double>         div_phi_u(dofs_per_cell);
  std::vector<Tensor<1, dim>> phi_u(dofs_per_cell);
  std::vector<Tensor<2, dim>> grad_phi_u(dofs_per_cell);
  std::vector<double>         phi_p(dofs_per_cell);

  double epsilonMMS = 0.001;

  double errorP = 0;
  double errorU = 0;
  double errorV = 0;
  
  double deltah = 0.05/std::pow(2.,number_of_initial_refinement-1); 
  int N_it = 0;
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);


      fe_values[velocities].get_function_values(solution, previous_velocity);

      fe_values[pressure].get_function_values(solution, previous_pressure);

      for (unsigned int q = 0; q < n_q_points; q++)
        {
              
          double x_q = fe_values.get_quadrature_points()[q][0];
          double x_q2 = x_q*x_q;
          double y_q = fe_values.get_quadrature_points()[q][1];
          double y_q2 = y_q*y_q;
          double sin2 = std::sin(x_q2 + y_q2);
          double cos2 = std::cos(x_q2 + y_q2);

          errorP = errorP + std::pow(abs(previous_pressure[q] - (sin2 + 2.)),2);
          errorU = errorU + std::pow(abs(previous_velocity[q][0] - (sin2 + epsilonMMS)),2);
          errorV = errorV + std::pow(abs(previous_velocity[q][1] - (cos2 + epsilonMMS)),2);
          ++N_it;
        }
    }
  
  errorP = std::sqrt(1./N_it*errorP);
  errorU = std::sqrt(1./N_it*errorU);
  errorV = std::sqrt(1./N_it*errorV);
  
  std::cout<<deltah<<std::endl;
  std::cout<<errorP<<std::endl;
  std::cout<<errorU<<std::endl;
  std::cout<<errorV<<std::endl;

  std::string filename = "errorL2-"+std::to_string(int(1.0 / viscosity)) +"-solution-" + Utilities::int_to_string(dim) + "d-" +
                         test_case + "-" + Utilities::int_to_string(number_of_initial_refinement) +
                         ".dat";
  std::ofstream outfile (filename);

  outfile<<deltah<<", "<<errorP<<", "<<errorU<<", "<<errorV<<std::endl;
}

template <int dim>
void
SteadyStateNavierStokes<dim>::solve_linear_system()
{
  std::cout << "    Solving system ..." << std::endl;

  // We use a direct solver for simplicity
  SparseDirectUMFPACK A_direct;
  A_direct.initialize(system_matrix);
  A_direct.vmult(delta_solution, system_rhs);
}

template <int dim>
void
SteadyStateNavierStokes<dim>::output_results(unsigned int iter) const
{
  std::cout << "    Output results ..." << std::endl;
  std::cout.precision(16);
  // For vector-valued problems we need to add the interpretation of the solution. 
  //  The dim first components are the velocity vectors and the following one is the pressure.
  std::vector<std::string> solution_names(dim, "velocity");
  solution_names.push_back("pressure");
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(
      dim, DataComponentInterpretation::component_is_part_of_vector);
  data_component_interpretation.push_back(
    DataComponentInterpretation::component_is_scalar);

  DataOut<dim> data_out;

  data_out.attach_dof_handler(this->dof_handler);

  // We attach the values as in the previous homeworks
  data_out.add_data_vector(solution,
                           solution_names,
                           DataOut<dim>::type_dof_data,
                           data_component_interpretation);

  data_out.build_patches();

  std::string filename = std::to_string(int(1.0 / viscosity)) +"-solution-" + Utilities::int_to_string(dim) + "d-" +
                         test_case + "-" + Utilities::int_to_string(number_of_initial_refinement) +
                         ".vtu";
  std::ofstream output(filename);
  data_out.write_vtu(output);


//   std::string filename2 = std::to_string(int(1.0 / viscosity)) +"-solution-" + Utilities::int_to_string(dim) + "d-" +
//                      test_case + "-" + Utilities::int_to_string(iter) +
//                      ".dat";
//   std::ofstream output2(filename2);
//   data_out.write_tecplot(output2);
}

template <int dim>
void
SteadyStateNavierStokes<dim>::run()
{
  std::cout << "---------------------------------------------------------"
            << std::endl;
  std::cout << "Solving steady state Navier-Stokes problem in " << dim
            << "D: " << test_case << std::endl;
  std::cout << "---------------------------------------------------------"
            << std::endl;

  setup_triangulation();
  setup_system();

  double alpha = 1;

  double error = 1e10;
  double errorMax = 1e5;
  double tol = 1e-10;
  int    iter  = 0;
  int    iterMax  = 80;
  
  Vector<double> evaluation_point;
  evaluation_point.reinit(dof_handler.n_dofs());

  // TODO: add newton loop and output the results at every iteration
  while(iter<iterMax && error>tol){

      // TODO: assemble, solve and update the solution here
      assemble_system();
      solve_linear_system();
      

      //Uncomment the next lines if you already calculated delta_solution
      std::cout << "iter: "<<iter +1  <<", "
                << "Linfty norm: "
                << delta_solution.linfty_norm()
                << " L2 norm: " << delta_solution.l2_norm() << std::endl;

      error = delta_solution.linfty_norm();

      
      // TODO: update solution and increase iterations
      solution.add(alpha , delta_solution);

      ++iter;
      
      // output results for each iteration
      
      if (error > errorMax)
      {
        std::cerr << std::endl
                  << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Solution diverge!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        break;
      }
    }

    output_results(iter);
    errorL2();
    // std::cout << std::setprecision (15) << solution[0]<< std::endl;
    // std::cout << std::setprecision (15) << solution[1]<< std::endl;
    // std::cout << std::setprecision (15) << solution[2]<< std::endl;
    // std::cout << std::setprecision (15) << solution[3]<< std::endl;
    // std::cout << std::setprecision (15) << solution[4]<< std::endl;
}


int
main(int argc, char *argv[])
{
  Settings parameters;
  if (!parameters.try_parse((argc > 1) ? (argv[1]) : ""))
    return 0;

  SteadyStateNavierStokes<2> navier_stokes_solver(parameters);
  navier_stokes_solver.run();

  return 0;
}