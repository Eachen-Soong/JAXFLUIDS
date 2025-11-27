from typing import Tuple, Union
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from jaxfluids.domain.domain_information import DomainInformation
from jaxfluids.time_integration.time_integrator import TimeIntegrator
from jaxfluids.stencils.spatial_derivative import SpatialDerivative
from jaxfluids.halos.halo_manager import HaloManager
from jaxfluids.data_types.numerical_setup.levelset import LevelsetExtensionSetup
from jaxfluids.stencils.levelset.first_deriv_first_order_center import FirstDerivativeFirstOrderCenter
from jaxfluids.stencils.levelset.first_deriv_second_order_center import FirstDerivativeSecondOrder
from jaxfluids.stencils.derivative.second_deriv_second_order_center import SecondDerivativeSecondOrderCenter
from jaxfluids.stencils.derivative.deriv_center_2 import DerivativeSecondOrderCenter
from jaxfluids.data_types.numerical_setup.levelset import NarrowBandSetup
from jaxfluids.config import precision

class QuantityExtender:
    """The QuantiyExtender extrapolates an arbitrary
    quantity in interface normal direction. Constant
    and linear extrapolation according to \cite Aslam2013
    is implemented.
    """


    def __init__(
            self,
            domain_information: DomainInformation,
            halo_manager: HaloManager,
            extension_setup: LevelsetExtensionSetup,
            narrowband_setup: NarrowBandSetup,
            extension_quantity: str = "primitives"
            ) -> None:

        if extension_quantity not in ("primitives", "interface"):
            raise RuntimeError

        self.eps = precision.get_eps()

        self.domain_information = domain_information
        self.halo_manager = halo_manager
        self.extension_setup = extension_setup
        self.narrowband_setup = narrowband_setup

        self.extension_quantity = extension_quantity
        nh_conservatives = self.domain_information.nh_conservatives
        nh_geometry = self.domain_information.nh_geometry
        if extension_quantity == "primitives":
            halos = nh_conservatives
        elif extension_quantity == "interface":
            halos = nh_geometry
        else:
            raise NotImplementedError

        time_integrator = extension_setup.time_integrator
        self.time_integrator: TimeIntegrator = time_integrator(
            nh = halos,
            inactive_axes = self.domain_information.inactive_axes)

        self.first_derivative_stencil = FirstDerivativeFirstOrderCenter(
            nh = halos,
            inactive_axes = self.domain_information.inactive_axes)
        
        self.spatial_stencil = extension_setup.spatial_stencil
        if self.spatial_stencil == "SECONDORDER":
            self.second_derivative_stencil = SecondDerivativeSecondOrderCenter(
                nh = halos,
                inactive_axes = self.domain_information.inactive_axes,
                offset = 1)

        halos_directional_derivative = 1 # TODO should be 2 for second order extrapolation
        assert_string = """Not enough halos for linear extrapolation."""
        assert nh_conservatives - halos_directional_derivative > 2, assert_string
        self.directional_derivative_stencil = FirstDerivativeSecondOrder(
            nh = halos,
            inactive_axes = self.domain_information.inactive_axes,
            offset = halos_directional_derivative)

        active_axes_indices = domain_information.active_axes_indices
        offset = nh_geometry - halos_directional_derivative
        self.s_0 = (...,) + tuple(
            [jnp.s_[offset:-offset] if
            i in active_axes_indices else
            jnp.s_[:] for i in range(3)]
            )
        
        offset = nh_conservatives - halos_directional_derivative
        self.s_1 = (...,) + tuple(
            [jnp.s_[offset:-offset] if
            i in active_axes_indices else
            jnp.s_[:] for i in range(3)]
            )

        self.s_2 = (...,) + tuple([
            jnp.s_[1:-1] if i in active_axes_indices
            else jnp.s_[:] for i in range(3) 
        ])

    def extend(
            self,
            quantity: Array,
            normal: Array,
            mask: Array,
            physical_simulation_time: float,
            CFL: float,
            steps: int,
            linear_extension: bool = False,
            debug: bool = False,
            force_steps: bool = False
            ) -> Tuple[Array, float]:
        """Extends the quantity in normal direction.

        :param quantity: _description_
        :type quantity: Array
        :param normal: _description_
        :type normal: Array
        :param mask: _description_
        :type mask: Array
        :param physical_simulation_time: _description_
        :type physical_simulation_time: float
        :param CFL: _description_
        :type CFL: float
        :param steps: _description_
        :type steps: int
        :param linear_extension: _description_, defaults to False
        :type linear_extension: bool, optional
        :param debug: _description_, defaults to False
        :type debug: bool, optional
        :return: _description_
        :rtype: Tuple[Array, float]
        """

        is_jaxforloop = self.extension_setup.is_jaxforloop
        is_jaxhileloop = self.extension_setup.is_jaxhileloop
        residual_threshold = self.extension_setup.residual_threshold
        is_parallel = self.domain_information.is_parallel

        smallest_cell_size = self.domain_information.smallest_cell_size
        fictitious_timestep_size = smallest_cell_size * CFL

        # Common body function for both fori_loop and while_loop
        def _scan_body_func_fori(carry, _) -> Tuple[Tuple[Array, Array], None]:
            current_quantity, current_directional_derivative = carry
            quantity_out, _ = self.do_integration_step(
                current_quantity, normal, mask, physical_simulation_time,
                fictitious_timestep_size, current_directional_derivative
            )
            return (quantity_out, current_directional_derivative), None

        def _scan_body_func_while(carry, _) -> Tuple[Tuple[Array, int, Array, Array], Tuple[Array, Array]]:
            current_quantity, current_index, _, current_directional_derivative = carry
            quantity_out, rhs = self.do_integration_step(
                current_quantity, normal, mask, physical_simulation_time,
                fictitious_timestep_size, current_directional_derivative
            )
            
            denominator = jnp.sum(mask, axis=(-1,-2,-3))
            mean_residual = jnp.sum(jnp.abs(rhs), axis=(-1,-2,-3))
            if is_parallel:
                mean_residual = jax.lax.psum(mean_residual, axis_name="i")
                denominator = jax.lax.psum(denominator, axis_name="i")
            mean_residual = mean_residual / (denominator + 1e-30) 
            mean_residual = jnp.mean(mean_residual) # NOTE mean over primitives and phases

            next_carry = (quantity_out, current_index + 1, mean_residual, current_directional_derivative)
            output = (quantity_out, mean_residual) # Output the quantity and residual for debugging/analysis
            return next_carry, output


        if is_jaxforloop:
            if linear_extension:
                directional_derivative = self.compute_directional_derivative(quantity, normal)
                buffer = jnp.zeros_like(quantity)
                directional_derivative = buffer.at[self.s_1].set(directional_derivative)
                
                # Perform scan for directional derivative extension
                initial_carry_dd = (directional_derivative, None)
                final_carry_dd, _ = jax.lax.scan(_scan_body_func_fori, initial_carry_dd, None, length=steps)
                directional_derivative = final_carry_dd[0]
            else:
                directional_derivative = None

            # Perform scan for quantity extension
            initial_carry_quantity = (quantity, directional_derivative)
            final_carry_quantity, _ = jax.lax.scan(_scan_body_func_fori, initial_carry_quantity, None, length=steps)
            quantity = final_carry_quantity[0]
            step_count = steps

        elif is_jaxhileloop:
            if linear_extension:
                directional_derivative = self.compute_directional_derivative(quantity, normal)
                buffer = jnp.zeros_like(quantity)
                directional_derivative = buffer.at[self.s_1].set(directional_derivative)
                
                # JAX's while_loop doesn't directly support `scan`-like accumulation of intermediate states for debugging easily.
                # If `debug` is true and you need intermediate steps for the directional derivative, you'd need to adapt.
                # For simplicity here, we're assuming debug for `linear_extension` is less critical for the intermediate steps.
                # If `force_steps` is True, it behaves like fori_loop, so we can use scan.
                # Otherwise, a custom while_loop-like structure for scan is more complex or not directly supported.
                # For `linear_extension` with `is_jaxhileloop`, if `force_steps` is False, the "steps" parameter is a limit, not a fixed number of iterations.
                # This makes it less directly convertible to a simple `scan` where the number of iterations is fixed.
                # For now, we'll keep the `fori_loop` structure for the directional derivative if `is_jaxhileloop` but `force_steps` is implied for its termination.
                # This part would need careful consideration if `debug` and dynamic steps were strict requirements here.
                initial_carry_dd = (directional_derivative, 0, 1e10, None)
                # Since the number of steps for directional_derivative extension is fixed by `steps`, we can use scan
                final_carry_dd, _ = jax.lax.scan(_scan_body_func_while, initial_carry_dd, None, length=steps)
                directional_derivative = final_carry_dd[0]
            else:
                directional_derivative = None

            # For the main quantity extension with jax.lax.while_loop, we need to manually create the loop
            # and potentially collect debug information. jax.lax.scan is generally for fixed iterations.
            # However, if `force_steps` is True, `while_loop` effectively becomes a fixed-step loop,
            # allowing for a `scan`-like approach for the main quantity extension.
            
            # If `force_steps` is True, we can use `scan` because the number of iterations is fixed.
            if force_steps:
                initial_carry_quantity = (quantity, 0, 1e10, directional_derivative)
                final_carry_quantity, _ = jax.lax.scan(_scan_body_func_while, initial_carry_quantity, None, length=steps)
                quantity = final_carry_quantity[0]
                step_count = final_carry_quantity[1]
            else:
                # If `force_steps` is False, we need the traditional jax.lax.while_loop structure.
                # Converting this directly to a single `scan` is not straightforward as `scan` expects a fixed length.
                # We can simulate `while_loop` behavior within a `scan` by using `jax.lax.cond` or `jax.lax.while_loop`
                # inside the scan body, but that's overly complex for this direct conversion.
                # For `while_loop` with a dynamic termination condition, a `scan` over a maximum number of steps
                # with conditional updates and then selecting the last valid state is one pattern.

                def _cond_fun_while(carry_args: Tuple[Array, int, float, Array]) -> bool:
                    _, index, mean_residual, _ = carry_args
                    condition1 = mean_residual > residual_threshold
                    condition2 = index < steps
                    return jnp.logical_and(condition1, condition2)

                def _body_func_while(carry_args: Tuple[Array, int, float, Array]) -> Tuple[Array, int, float, Array]:
                    current_quantity, current_index, _, current_directional_derivative = carry_args
                    quantity_out, rhs = self.do_integration_step(
                        current_quantity, normal, mask, physical_simulation_time,
                        fictitious_timestep_size, current_directional_derivative
                    )
                    denominator = jnp.sum(mask, axis=(-1,-2,-3))
                    mean_residual = jnp.sum(jnp.abs(rhs), axis=(-1,-2,-3))
                    if is_parallel:
                        mean_residual = jax.lax.psum(mean_residual, axis_name="i")
                        denominator = jax.lax.psum(denominator, axis_name="i")
                    mean_residual = mean_residual/(denominator + 1e-30) 
                    mean_residual = jnp.mean(mean_residual) # NOTE mean over primitives and phases
                    return (quantity_out, current_index + 1, mean_residual, current_directional_derivative)

                initial_carry_quantity = (quantity, 0, 1e10, directional_derivative)
                final_carry_quantity = jax.lax.while_loop(_cond_fun_while, _body_func_while, initial_carry_quantity)
                quantity = final_carry_quantity[0]
                step_count = final_carry_quantity[1]


        return quantity, step_count

    def compute_residual(
            self,
            quantity: Array,
            normal: Array,
            mask: Array
            ) -> Tuple[float, float]:
        """Computes the mean and max residual.

        :param quantity: _description_
        :type quantity: Array
        :param mask: _description_
        :type mask: Array
        :return: _description_
        :rtype: _type_
        """
        is_parallel = self.domain_information.is_parallel
        rhs = self.compute_rhs(quantity, normal, mask)
        residual = jnp.abs(rhs)
        mean_residual = jnp.sum(residual, axis=(-3,-2,-1))
        denominator = jnp.sum(mask, axis=(-3,-2,-1))
        max_residual = jnp.max(residual)
        if is_parallel:
            mean_residual = jax.lax.psum(mean_residual, axis_name="i")
            denominator = jax.lax.psum(denominator, axis_name="i")
            max_residual = jax.lax.pmax(max_residual, axis_name="i")
        mean_residual = mean_residual/(denominator + 1e-30)
        mean_residual = jnp.mean(mean_residual) # NOTE mean over primitives and phases
        return mean_residual, max_residual, rhs


    def compute_directional_derivative(
            self,
            quantity: Array,
            normal: Array,
            ) -> Array:
        """Computes the directional derivative 
        required for linear extrapolation.

        :param quantity: _description_
        :type quantity: Array
        :param normal: _description_
        :type normal: Array
        :return: _description_
        :rtype: Array
        """
        cell_size = self.domain_information.smallest_cell_size
        active_axes_indices = self.domain_information.active_axes_indices
        directional_derivative = 0.0
        for axis in active_axes_indices:
            normal_xi = normal[axis][self.s_0]
            deriv_xi_L = self.directional_derivative_stencil.derivative_xi(quantity, cell_size, axis, 0)
            deriv_xi_R = self.directional_derivative_stencil.derivative_xi(quantity, cell_size, axis, 1)
            mask_L = jnp.where(normal_xi >= 0.0, 1, 0)
            mask_R = 1 - mask_L
            derv_xi = deriv_xi_L * mask_L + deriv_xi_R * mask_R
            directional_derivative += derv_xi * normal_xi
        return directional_derivative


    @partial(jax.jit, static_argnums=(0))
    def do_integration_step(
            self,
            quantity: Array,
            normal: Array,
            mask: Array,
            physical_simulation_time: float,
            fictitious_timestep_size: float,
            directional_derivative: Array = None
            ) -> Array:
        """Performs an integration step of the extension equation.

        :param quantity: Quantity buffer
        :type quantity: Array
        :param normal: Normal buffer
        :type normal: Array
        :param mask: Mask indicating where to extend
        :type mask: Array
        :param fictitious_timestep_size: Fictitious time step size
        :type fictitious_timestep_size: float
        :return: Integrated quantity buffer and corresponding right-hand-side buffer
        :rtype: Tuple[Array, Array]
        """

        if self.time_integrator.no_stages > 1:
            init = jnp.array(quantity, copy=True)
        for stage in range( self.time_integrator.no_stages ):
            rhs = self.compute_rhs(quantity, normal, mask, directional_derivative)
            if stage > 0:
                quantity = self.time_integrator.prepare_buffer_for_integration(quantity, init, stage)
            quantity = self.time_integrator.integrate(quantity, rhs, fictitious_timestep_size, stage)
            if self.extension_quantity == "primitives":
                quantity = self.halo_manager.perform_halo_update_material(
                    quantity, physical_simulation_time, False, False)
            elif self.extension_quantity == "interface":
                quantity = self.halo_manager.perform_halo_update_levelset(
                    quantity, False, False, is_geometry_halos=True)
        return quantity, rhs

    def compute_rhs(
            self,
            quantity: Array,
            normal: Array,
            mask: Array,
            directional_derivative: Array = None
            ) -> Array:
        """Computes the right-hand-side of the exension equation.

        :param quantity: Quantity buffer
        :type quantity: Array
        :param normal: Normal buffer
        :type normal: Array
        :param mask: Mask indiciating where to extend
        :type mask: Array
        :return: Right-hand-side of the extension equation
        :rtype: Array
        """


        nhx, nhy, nhz = self.domain_information.domain_slices_conservatives
        nhx_, nhy_, nhz_ = self.domain_information.domain_slices_geometry
        dxi = self.domain_information.smallest_cell_size
        active_axes_indices = self.domain_information.active_axes_indices

        rhs = 0.0
        for axis in active_axes_indices:
            deriv_L = self.first_derivative_stencil.derivative_xi(quantity, dxi, axis, 0)
            deriv_R = self.first_derivative_stencil.derivative_xi(quantity, dxi, axis, 1)

            if self.spatial_stencil == "SECONDORDER":
                second_deriv = self.second_derivative_stencil.derivative_xi(quantity, dxi, axis)
                second_deriv_L = self.minmod(second_deriv, jnp.roll(second_deriv, 1, axis))
                second_deriv_R = self.minmod(second_deriv, jnp.roll(second_deriv, -1, axis))
                deriv_L = deriv_L + dxi/2.0 * second_deriv_L[self.s_2]
                deriv_R = deriv_R - dxi/2.0 * second_deriv_R[self.s_2]

            # UPWINDING
            mask_L = jnp.where(normal[axis,...,nhx_,nhy_,nhz_] >= 0.0, 1.0, 0.0)
            mask_R = 1.0 - mask_L

            rhs -= normal[axis,...,nhx_,nhy_,nhz_] * (mask_L * deriv_L + mask_R * deriv_R)

        if directional_derivative != None:
            rhs += directional_derivative[...,nhx,nhy,nhz]

        rhs *= mask

        return rhs


    def minmod(
            self,
            array1: Array,
            array2: Array
            ) -> Array:
        """Minmod limiter

        :param array1: _description_
        :type array1: Array
        :param array2: _description_
        :type array2: Array
        :return: _description_
        :rtype: Array
        """
        mask = jnp.where(array1*array2 < 0.0, 0, 1)
        out = jnp.minimum(jnp.abs(array1), jnp.abs(array2))
        out *= mask
        return out