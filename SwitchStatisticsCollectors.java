package net.floodlightcontroller.mlids;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Map;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;

import org.projectfloodlight.openflow.protocol.match.Match;
import org.projectfloodlight.openflow.types.DatapathId;

import javafx.util.Pair;
import net.floodlightcontroller.core.IFloodlightProviderService;
import net.floodlightcontroller.core.module.FloodlightModuleContext;
import net.floodlightcontroller.core.module.FloodlightModuleException;
import net.floodlightcontroller.core.module.IFloodlightModule;
import net.floodlightcontroller.core.module.IFloodlightService;
import net.floodlightcontroller.threadpool.IThreadPoolService;
import net.floodlightcontroller.statistics.FlowRuleStats;
import net.floodlightcontroller.statistics.StatisticsCollector;

public class SwitchStatisticsCollectors implements IFloodlightModule {

	private static IThreadPoolService threadPoolService;
	private static ScheduledFuture<?> flowStatsCollector;
	private static int flowStatsInterval = 10;
	private static StatisticsCollector stats_collector;
	@Override
	public Collection<Class<? extends IFloodlightService>> getModuleServices() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Map<Class<? extends IFloodlightService>, IFloodlightService> getServiceImpls() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Collection<Class<? extends IFloodlightService>> getModuleDependencies() {
		Collection<Class<? extends IFloodlightService>> l = new ArrayList<Class<? extends IFloodlightService>>();
		l.add(IFloodlightProviderService.class);
		l.add(IThreadPoolService.class);
		return l;
	}

	@Override
	public void init(FloodlightModuleContext context) throws FloodlightModuleException {
		// TODO Auto-generated method stub
		stats_collector = new StatisticsCollector();
		threadPoolService = context.getServiceImpl(IThreadPoolService.class);
	}

	@Override
	public void startUp(FloodlightModuleContext context) throws FloodlightModuleException {
		// TODO Auto-generated method stub
		flowStatsCollector = threadPoolService.getScheduledExecutor()
				.scheduleAtFixedRate( new getFlowStatistics(),flowStatsInterval, flowStatsInterval, TimeUnit.SECONDS);
	}

	private class getFlowStatistics implements Runnable {
		@Override
		public void run()  {
			Map<Pair<Match, DatapathId>, FlowRuleStats> flowstats = stats_collector.getFlowStats();
			System.out.println("Flow statistics collecting...");
			for(Pair<Match, DatapathId> s: flowstats.keySet())
			{
				FlowRuleStats stats = flowstats.get(s);
				System.out.println(s);
				System.out.println(stats.getDurationSec() + " - " + stats.getHardTimeout() + " - " + stats.getIdleTimeout() + " - " + stats.getByteCount().getValue() + " - " + stats.getPacketCount().getValue());
			}
			
		}
	}

}
